import argparse
import os
import glob
import h5py
import random
import numpy as np
import hashlib
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="Preprocessing images script")

parser.add_argument('-sample_size', type=int, default=1000,
                    help="Number of images used to calculate the median threshold (if blur filtering enabled)")
parser.add_argument('-blur_threshold_factor', type=int, default=100,
                    help="Factor subtracted from the median variance of Laplacian (if blur filtering enabled)")
parser.add_argument('-image_size', type=int, default=(224, 224), nargs=2,
                    help="The size to which all images are resized")

parser.add_argument('-input_dir', type=str, default=r'C:\Users\Intel User\Desktop\Downy',
                    help="Directory where the input images are located")
parser.add_argument('-output_dir', type=str, default=r'C:\Users\Intel User\Desktop\Downy',
                    help="Directory where the output HDF5 files are saved")

parser.add_argument('-k_fold', action='store_true',
                    help="Perform k-fold cross validation (train/val folds) with a single 10% holdout test")
parser.add_argument('-k_fold_number', type=int, default=5,
                    help="Number of folds for k-fold cross validation")

parser.add_argument('--use_existing_hdf5', action='store_true',
                    help="Load one or more existing HDF5 files and combine them before splitting.")
parser.add_argument('--balance_classes', action='store_true',
                    help="Downsample each class to match the smallest class (after cleaning).")

parser.add_argument('--hdf5_files', type=str, nargs='*', default=None,
                    help="Explicit list of HDF5 files to load (overrides --hdf5_dir).")
parser.add_argument('--hdf5_dir', type=str, default=None,
                    help="Directory to scan for .hdf5 if --hdf5_files not given.")
parser.add_argument('--dedupe_loaded', action='store_true',
                    help="Run duplicate removal after loading HDF5s (hash-based).")

parser.add_argument('--dedupe_by', type=str, default='image',
                    choices=['image', 'image+label'],
                    help="Duplicate definition: 'image' ignores label; 'image+label' treats same image with "
                         "different labels as distinct.")

# NEW: label parsing
parser.add_argument('--label_map', type=str, nargs='*', default=None,
                    help="Optional explicit label mapping like: clear=0 infected=1 conidiophores=2. "
                         "If omitted, labels are auto-mapped from filenames.")
parser.add_argument('--label_sep', type=str, default='_',
                    help="Separator to split filename for label extraction (default: '_'). "
                         "Label is taken from last token before extension.")
parser.add_argument('--skip_test_set', action='store_true', help="Skip creating a test set.")

args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

sample_size = args.sample_size
blur_threshold_factor = args.blur_threshold_factor
image_size = tuple(args.image_size)
input_dir = args.input_dir
output_dir = args.output_dir


# -------------------------
# Helpers
# -------------------------
def parse_label_map(label_map_items):
    """
    Parse ['clear=0','infected=1'] -> {'clear':0,'infected':1}
    """
    if not label_map_items:
        return None
    out = {}
    for item in label_map_items:
        if '=' not in item:
            raise ValueError(f"--label_map items must be like name=int, got: {item}")
        k, v = item.split('=', 1)
        k = k.strip().lower()
        v = int(v.strip())
        out[k] = v
    return out


EXPLICIT_LABEL_MAP = parse_label_map(args.label_map)
AUTO_LABEL_MAP = {}  # filled if EXPLICIT_LABEL_MAP is None


def extract_label_from_filename(filename, sep='_'):
    """
    label = last token after splitting by sep, before extension.
    Example: img_123_clear.png -> 'clear'
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split(sep)
    return parts[-1].lower().strip()


def label_to_int(label_str):
    """
    Map label string to int using explicit map if given,
    otherwise build an auto map.
    """
    if EXPLICIT_LABEL_MAP is not None:
        if label_str not in EXPLICIT_LABEL_MAP:
            raise ValueError(f"Label '{label_str}' not found in --label_map {EXPLICIT_LABEL_MAP}")
        return EXPLICIT_LABEL_MAP[label_str]

    # auto-map
    if label_str not in AUTO_LABEL_MAP:
        AUTO_LABEL_MAP[label_str] = len(AUTO_LABEL_MAP)
    return AUTO_LABEL_MAP[label_str]


def downsample_to_min_class(images, labels, seed=42):
    """
    Downsample each class to the smallest class count.
    labels can be (N,) or (N,1). Returns images, labels shaped (N,1).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("Cannot balance classes: labels array is empty (no samples loaded).")

    if labels.ndim == 2 and labels.shape[1] == 1:
        y = labels[:, 0]
    elif labels.ndim == 1:
        y = labels
        labels = labels.reshape(-1, 1)
    else:
        raise ValueError(f"labels must be (N,) or (N,1); got {labels.shape}")

    classes, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        raise ValueError("Cannot balance classes: no classes found (empty labels after processing).")

    min_count = int(counts.min())
    if min_count == 0:
        raise ValueError(
            f"Cannot balance classes: at least one class has 0 samples. counts={dict(zip(classes, counts))}")

    rng = np.random.default_rng(seed)
    keep_indices = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        keep_indices.append(chosen)

    keep_indices = np.concatenate(keep_indices)
    rng.shuffle(keep_indices)

    images_bal = images[keep_indices]
    labels_bal = labels[keep_indices].astype(np.int64, copy=False)

    print(f"Balancing classes to {min_count} samples per class")
    for cls, c in zip(classes, counts):
        print(f" - class {cls}: {min_count} (was {int(c)})")
    print('..................................................')

    return images_bal, labels_bal


def find_duplicate_images_and_labels(images, labels, dedupe_by='image'):
    """
    Remove duplicates by hashing each image array.

    dedupe_by:
      - 'image' (recommended): image bytes define uniqueness, label ignored
      - 'image+label': (image hash, label) define uniqueness

    Returns: (duplicates, images_nd, labels_nd) where labels_nd has shape (N,1)
    """
    image_dict = {}
    duplicates = []
    non_dup_images = []
    non_dup_labels = []

    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    elif labels.ndim == 2 and labels.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Labels must be (N,) or (N,1). Got {labels.shape}")
    labels = labels.astype(np.int64, copy=False)

    for i in range(len(images)):
        img = np.ascontiguousarray(images[i])

        h = hashlib.md5()
        h.update(str(img.shape).encode())
        h.update(str(img.dtype).encode())
        h.update(img.tobytes())
        img_hash = h.hexdigest()

        label_scalar = int(labels[i, 0])
        key = (img_hash, label_scalar) if dedupe_by == 'image+label' else img_hash

        if key in image_dict:
            duplicates.append((i, image_dict[key]))
            continue

        image_dict[key] = i
        non_dup_images.append(img)
        non_dup_labels.append(label_scalar)

    print(f'Number of duplicates found: {len(duplicates)}')
    print('..................................................')

    return duplicates, np.array(non_dup_images), np.array(non_dup_labels, dtype=np.int64).reshape(-1, 1)


def load_hdf5_datasets(paths):
    imgs_list, labels_list = [], []
    for p in paths:
        with h5py.File(p, 'r') as f:
            if 'images' not in f or 'labels' not in f:
                raise KeyError(f"{p} missing 'images' or 'labels' dataset.")
            imgs = f['images'][:]
            lbls = f['labels'][:]
            imgs_list.append(imgs)
            labels_list.append(lbls)
            print(f"Loaded {imgs.shape[0]} samples from {os.path.basename(p)}")

    images = np.concatenate(imgs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    elif labels.ndim == 2 and labels.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Labels must be shape (N,) or (N,1), got {labels.shape}")

    labels = labels.astype(np.int64, copy=False)
    return images, labels


def gather_hdf5_paths(hdf5_files, hdf5_dir, default_dir):
    """Resolve list of .hdf5 files to load."""
    if hdf5_files:
        return hdf5_files

    search_dir = hdf5_dir or default_dir
    if not search_dir:
        raise ValueError("Provide --hdf5_files or --hdf5_dir or ensure --output_dir is set.")

    candidates = []
    patterns = [
        os.path.join(search_dir, "train_*.hdf5"),
        os.path.join(search_dir, "val_*.hdf5"),
        os.path.join(search_dir, "test_*.hdf5"),
        os.path.join(search_dir, "train.hdf5"),
        os.path.join(search_dir, "val.hdf5"),
        os.path.join(search_dir, "test.hdf5"),
        os.path.join(search_dir, "*.hdf5"),
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in seen:
                candidates.append(p)
                seen.add(p)

    if not candidates:
        raise FileNotFoundError(f"No .hdf5 files found in {search_dir}")
    return candidates


# -------------------------
# Load data
# -------------------------
if args.use_existing_hdf5:
    print("Loading datasets from existing HDF5 files...")
    hdf5_paths = gather_hdf5_paths(args.hdf5_files, args.hdf5_dir, output_dir)
    print("Files to combine:")
    for p in hdf5_paths:
        print(" -", p)

    images, labels = load_hdf5_datasets(hdf5_paths)

    if args.dedupe_loaded:
        print('Removing duplicate images across loaded HDF5 datasets')
        _, images, labels = find_duplicate_images_and_labels(images, labels, dedupe_by=args.dedupe_by)

else:
    print("Resizing and converting to RGB")
    print('..................................................')

    images_list = []
    labels_list = []
    skipped = 0
    considered = 0

    for filename in os.listdir(input_dir):
        fname_lower = filename.lower()
        if not fname_lower.endswith(('.png', '.jpg', '.jpeg')):
            continue

        considered += 1
        try:
            with Image.open(os.path.join(input_dir, filename)) as img:
                img = img.resize(image_size).convert('RGB')
                img_arr = np.array(img)

            label_str = extract_label_from_filename(filename, sep=args.label_sep)
            label_num = label_to_int(label_str)

            images_list.append(img_arr)
            labels_list.append(label_num)

        except Exception as e:
            skipped += 1
            print(f"Skipping {filename}: {e}")

    if considered == 0:
        raise FileNotFoundError(f"No .png/.jpg/.jpeg files found in {input_dir}")

    if len(images_list) == 0:
        raise ValueError(
            f"Loaded 0 images from {input_dir}. "
            f"Check filename label format and/or --label_map. "
            f"Considered {considered}, skipped {skipped}."
        )

    images = np.array(images_list)
    labels = np.array(labels_list)  # (N,)

    if EXPLICIT_LABEL_MAP is None:
        print("Auto label mapping (from filenames):")
        for k, v in sorted(AUTO_LABEL_MAP.items(), key=lambda kv: kv[1]):
            print(f" - {k} => {v}")
        print('..................................................')

    print('Removing duplicate images')
    _, images, labels = find_duplicate_images_and_labels(images, labels, dedupe_by=args.dedupe_by)

# -------------------------
# Optional blur filtering would go here (after dedupe, before balancing)
# -------------------------

# -------------------------
# Optional class balancing (after cleaning)
# -------------------------
if args.balance_classes:
    images, labels = downsample_to_min_class(images, labels, seed=SEED)

# -------------------------
# Count labels
# -------------------------
y1d_all = labels.reshape(-1)
print('Counting labels')
classes, counts = np.unique(y1d_all, return_counts=True)
for cls, c in zip(classes, counts):
    print(f"Number of class {cls} labels: {int(c)}")
print('..................................................')

# -------------------------
# Splitting & saving
# -------------------------
# -------------------------
# Splitting & saving
# -------------------------
os.makedirs(output_dir, exist_ok=True)


def write_h5(path, X, y_1d):
    y = np.asarray(y_1d, dtype=np.int64).reshape(-1, 1)
    with h5py.File(path, 'w') as f:
        f.create_dataset('images', data=X)
        f.create_dataset('labels', data=y)


if args.k_fold:
    k_folds = args.k_fold_number

    if args.skip_test_set:
        # No holdout test. Do k-fold directly on all data.
        X_dev = images
        y_dev_1d = y1d_all
        X_test = None
        y_test = None

        print("[k-fold] skip_test_set enabled: no holdout test set will be created.")
        print(f"[k-fold] Dev (all data): {X_dev.shape}, {y_dev_1d.shape}")
        print('..................................................')
    else:
        # 1) Holdout 10% test set (stratified)
        X_dev, X_test, y_dev_1d, y_test_1d = train_test_split(
            images, y1d_all,
            test_size=0.10,
            stratify=y1d_all,
            random_state=SEED
        )

        # Save holdout test once
        write_h5(os.path.join(output_dir, 'test.hdf5'), X_test, y_test_1d)

        print(f"[Holdout] Dev:  {X_dev.shape}, {y_dev_1d.shape}")
        print(f"[Holdout] Test: {X_test.shape}, {np.asarray(y_test_1d).reshape(-1, 1).shape}")
        print('..................................................')

    # sanity check: each class in dev must have >= k samples
    uniq, cnts = np.unique(y_dev_1d, return_counts=True)
    for cls, c in zip(uniq, cnts):
        if c < k_folds:
            raise ValueError(
                f"Class {cls} has only {c} samples in the dev set, need >= n_splits={k_folds}. "
                f"Try lowering --k_fold_number or disabling balancing."
            )

    # 2) k-fold train/val on dev
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev_1d)):
        X_train, X_val = X_dev[train_idx], X_dev[val_idx]
        y_train_1d, y_val_1d = y_dev_1d[train_idx], y_dev_1d[val_idx]

        print(f"[Fold {fold}] Train: {X_train.shape}, {np.asarray(y_train_1d).reshape(-1, 1).shape}")
        print(f"[Fold {fold}] Val:   {X_val.shape}, {np.asarray(y_val_1d).reshape(-1, 1).shape}")

        if X_test is not None:
            print(
                f"[Fold {fold}] Holdout Test (constant): {X_test.shape}, {np.asarray(y_test_1d).reshape(-1, 1).shape}")
        print('..................................................')

        write_h5(os.path.join(output_dir, f'train_{fold}.hdf5'), X_train, y_train_1d)
        write_h5(os.path.join(output_dir, f'val_{fold}.hdf5'), X_val, y_val_1d)

else:
    if args.skip_test_set:
        # Single split: 90/10 (train/val), stratified
        X_train, X_val, y_train_1d, y_val_1d = train_test_split(
            images, y1d_all,
            test_size=0.111,
            stratify=y1d_all,
            random_state=SEED
        )

        print("[single split] skip_test_set enabled: creating train/val only (90/10).")
        print(f"Training set shape:   {X_train.shape}, {np.asarray(y_train_1d).reshape(-1, 1).shape}")
        print(f"Validation set shape: {X_val.shape}, {np.asarray(y_val_1d).reshape(-1, 1).shape}")
        print('..................................................')

        write_h5(os.path.join(output_dir, 'train.hdf5'), X_train, y_train_1d)
        write_h5(os.path.join(output_dir, 'val.hdf5'), X_val, y_val_1d)

    else:
        # Single split: 80/10/10 (train/test/val) via two-step stratified split
        X_train, X_temp, y_train_1d, y_temp_1d = train_test_split(
            images, y1d_all,
            test_size=0.20,
            stratify=y1d_all,
            random_state=SEED
        )

        X_test, X_val, y_test_1d, y_val_1d = train_test_split(
            X_temp, y_temp_1d,
            test_size=0.50,
            stratify=y_temp_1d,
            random_state=SEED
        )

        print(f"Training set shape:   {X_train.shape}, {np.asarray(y_train_1d).reshape(-1, 1).shape}")
        print(f"Testing set shape:    {X_test.shape}, {np.asarray(y_test_1d).reshape(-1, 1).shape}")
        print(f"Validation set shape: {X_val.shape}, {np.asarray(y_val_1d).reshape(-1, 1).shape}")
        print('..................................................')

        write_h5(os.path.join(output_dir, 'train.hdf5'), X_train, y_train_1d)
        write_h5(os.path.join(output_dir, 'test.hdf5'), X_test, y_test_1d)
        write_h5(os.path.join(output_dir, 'val.hdf5'), X_val, y_val_1d)
