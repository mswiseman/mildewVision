import os
import h5py
import glob
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn.functional as F
import torchvision.transforms as tvtrans


np.random.seed(2020)


def printArgs(logger, args):
    for k, v in args.items():
        if logger:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


def load_f5py(dataset_para):
    """
        Load data from HDF5 files or image directory
    """
    f = h5py.File(dataset_para['dataset_folder'] /
                  dataset_para['test_filepath'], 'r')
    image_ds = f['images']
    images = image_ds[:, ]
    label_ds = f['labels']
    labels = label_ds[:]
    return images, labels


def load_dir(dataset_para):
    label_class_map = {'clear': 0, 'infected': 1}

    image_folder = dataset_para['image_folder']
    image_filenames = glob.glob(str(image_folder / '*.png'))
    num = len(image_filenames)
    images = np.ndarray(shape=(num, 224, 224, 3), dtype=np.uint8)
    labels = np.zeros(shape=(num, 1), dtype=np.uint8)
    image_filename_list = []

    for i, image_filename in enumerate(image_filenames):
        # Get labels if possible
        image_filename = os.path.basename(image_filename)
        image_filename_text = os.path.splitext(image_filename)[0]
        filename_strs = image_filename_text.split('_')
        # Determine classified or not
        if filename_strs[-1] in list(label_class_map.keys()):
            labels[i] = label_class_map[filename_strs[-1]]

        image_filepath = image_folder / image_filename
        img = Image.open(image_filepath)
        images[i] = np.asarray(img)
        image_filename_list.append(image_filename_text)

    return images, image_filename_list, labels


def pred_img(img, model):
    """
        Get predicted image class and prob using well-trained model
    Args:
        img: PIL image or np.ndarray
    """

    out = model(img)

    pred = torch.argmax(out, axis=1)
    prob = F.softmax(out, dim=1)

    return pred, prob


def categorize(pred, true, idx, t_p, t_n, f_p, f_n):
    """
        Categorize each predicted result into one of four categories in a confusion matrix
        NOTE: This is only meaningful for binary classification where class 1 is "positive"
              and class 0 is "negative".
    """
    status = "Correct"
    if pred == 0:
        if pred == true:
            t_n.append(idx)
        else:
            f_n.append(idx)
            status = "Incorrect"
    else:
        if pred == true:
            t_p.append(idx)
        else:
            f_p.append(idx)
            status = "Incorrect"

    return status


def _ensure_1d_labels(labels: np.ndarray) -> np.ndarray:
    """Make labels shape (N,) int64 regardless of input being (N,), (N,1), etc."""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels.astype(np.int64, copy=False)


def _default_class_name_map(num_classes: int) -> dict[int, str]:
    """Fallback naming if you don't have explicit class names for >2 classes."""
    if num_classes == 2:
        return {0: "clear", 1: "infected"}
    if num_classes == 3:
        return {0: "clear", 1: "infected", 2: "conidiophores"}
    return {i: f"class_{i}" for i in range(num_classes)}

def save_misclassified_montage(misclassified, class_name_map, out_path, n=10, seed=2020, overlay=True):
    if len(misclassified) == 0:
        print("No misclassified samples to plot.")
        return

    rng = np.random.default_rng(seed)
    n = min(n, len(misclassified))
    picks = rng.choice(len(misclassified), size=n, replace=False)

    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= n:
            ax.axis("off")
            continue

        ex = misclassified[picks[ax_i]]
        img = ex["img"]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)

        # CHW -> HWC if needed
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # grayscale -> 2D
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        ax.imshow(img)

        t = class_name_map.get(ex["true"], str(ex["true"]))
        p = class_name_map.get(ex["pred"], str(ex["pred"]))

        pred_prob = ex.get("pred_prob", None)
        true_prob = ex.get("true_prob", None)

        if overlay and pred_prob is not None and true_prob is not None:
            ax.text(
                0.02, 0.98,
                f"P(pred)={pred_prob:.2f}\nP(true)={true_prob:.2f}",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )
            ax.set_title(f"id {ex['idx']}\ntrue: {t} → pred: {p}", fontsize=10)

        else:
            if pred_prob is not None and true_prob is not None:
                ax.set_title(
                    f"id {ex['idx']}\ntrue: {t} → pred: {p}\n"
                    f"P(pred)={pred_prob:.2f}, P(true)={true_prob:.2f}",
                    fontsize=9
                )
            else:
                ax.set_title(f"id {ex['idx']}\ntrue: {t} → pred: {p}", fontsize=10)

        ax.axis("off")

    fig.suptitle(f"Random misclassified samples (n={n})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved misclassified montage: {out_path}")


if __name__ == "__main__":
    from utils import init_model, load_model, parse_model, plot_confusion_matrix

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='ResNet', help='model used for training')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model parameters')
    parser.add_argument('--loading_epoch', type=int, required=True, help='xth model loaded for inference')
    parser.add_argument('--timestamp', required=True, help='model timestamp')
    parser.add_argument('--outdim', type=int, default=2, help='number of classes')

    parser.add_argument('--cuda', action='store_true', help='enable cuda')
    parser.add_argument('--cuda_id', default="1", help='specify cuda')
    parser.add_argument('--mps', action='store_true', help='enable Apple MPS (Metal) backend')

    parser.add_argument('--img_folder', type=str, default='images', help='image folder')
    parser.add_argument('--dataset_path', type=str, required=True, help='path to data')
    parser.add_argument('--model_path', type=str, required=True, help='path to model')

    # Better CLI pattern: flags should be store_true/store_false, not type=bool
    parser.add_argument('--HDF5', action='store_true', help='load from HDF5 (if omitted, loads from image directory)')

    parser.add_argument('--group', type=str, default='baseline', help='exp group')
    parser.add_argument('--cv_dai', type=str, help='date to be tested')
    parser.add_argument('--cv_qtl', type=str, help='qtl partition to be tested')
    parser.add_argument('--cv_seg_dataset', type=str, help='seg dataset to be tested')
    parser.add_argument('--save_misclassified', action='store_true',
                        help='save a montage of randomly selected misclassified images')
    parser.add_argument('--n_misclassified', type=int, default=10,
                        help='number of misclassified images to save (max)')

    parser.add_argument('--set', default='val', choices=['train', 'val', 'test'],
                        help='use train/val/test set for inference')

    parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361],
                        help='List of means for each channel')
    parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192],
                        help='List of standard deviations for each channel')

    # Optional: quick toggle for AMP during inference (mostly helpful on CUDA)
    parser.add_argument('--amp', action='store_true', help='enable autocast during inference (CUDA/MPS)')

    opt = parser.parse_args()

    # --- Paths / dataset selection ---
    model_para = parse_model(opt)

    dataset_root_path = Path(opt.dataset_path) / 'data'
    image_folder = dataset_root_path / opt.img_folder

    output_folder = Path(os.getcwd()).parent / 'results' / 'journal'

    means = opt.means
    stds = opt.stds

    subfolder_name = 'inference_results'
    if opt.set == 'train':
        test_filepath = 'train.hdf5'
    elif opt.set == 'val':
        test_filepath = 'val.hdf5'
        if opt.cv_dai:
            dataset_root_path = dataset_root_path / 'cross_validation_ds' / opt.cv_dai
            subfolder_name = f'cross_validation/{opt.cv_dai}'
        if opt.cv_qtl:
            dataset_root_path = dataset_root_path / 'qtl_partition_test' / f'partition_{opt.cv_qtl}'
            subfolder_name = f'cross_validation/partition_{opt.cv_qtl}'
        if opt.cv_seg_dataset:
            dataset_root_path = dataset_root_path / 'segmentation' / 'cls_dataset' / f'cv{opt.cv_seg_dataset}'
            subfolder_name = f'cross_validation/seg_dataset_{opt.cv_seg_dataset}'
    else:  # test
        test_filepath = 'test_set.hdf5'

    output_folder = output_folder / subfolder_name / opt.group / opt.model_type
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_para = {
        'dataset_folder': dataset_root_path,
        'test_filepath': test_filepath,
        'image_folder': image_folder
    }

    printArgs(None, vars(opt))

    # --- Load data ---
    if opt.HDF5:
        images, labels = load_f5py(dataset_para)
        image_filenames = None
    else:
        images, image_filenames, labels = load_dir(dataset_para)

    labels_1d = _ensure_1d_labels(labels)

    # --- Transforms ---
    # If your HDF5 images are uint8 HWC arrays, ToTensor will handle them.
    # For safety, ensure RGB and correct type when reading from different sources.
    if opt.model_type == 'Inception3':
        test_transform = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.Resize(299),
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])
    else:
        test_transform = tvtrans.Compose([
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])

    # --- Load model ---
    model, device = load_model(model_para)
    model.eval()

    if opt.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{opt.cuda_id}")
    elif opt.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(f"Using device: {device}")

    # --- Output records ---
    META_COL_NAMES = ['id', 'predicted class', 'true class', 'status']
    rows = []  # collect dicts/rows and build DataFrame once

    num_classes = int(opt.outdim)
    class_name_map = _default_class_name_map(num_classes)

    # Metrics accumulators
    y_true: list[int] = []
    y_pred: list[int] = []

    # Binary-only confusion breakdown lists (only meaningful if num_classes == 2)
    f_n, f_p, t_n, t_p = [], [], [], []
    misclassified = []  # list of dicts: {idx, img, true, pred}

    print("INFERENCE START")

    correct_counts = 0
    total_counts = int(images.shape[0])

    amp_enabled = bool(opt.amp) and (device.type == "cuda")

    with torch.no_grad():
        for idx in range(total_counts):
            cur_img = images[idx]

            # Some datasets might store grayscale; ensure 3 channels if needed (optional).
            # If you know all images are RGB already, you can remove this.
            if isinstance(cur_img, np.ndarray) and cur_img.ndim == 2:
                cur_img = np.stack([cur_img] * 3, axis=-1)

            preproc_img = test_transform(cur_img).unsqueeze(0).to(device)

            true_label = int(labels_1d[idx])
            y_true.append(true_label)

            # Inference (optional autocast)
            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    pred_t, prob_t = pred_img(preproc_img, model)
            else:
                pred_t, prob_t = pred_img(preproc_img, model)

            pred_label = int(pred_t.cpu().item())
            y_pred.append(pred_label)

            correct = (pred_label == true_label)
            correct_counts += int(correct)

            pred_prob = float(prob_t.squeeze(0)[pred_label].detach().cpu().item())
            true_prob = float(prob_t.squeeze(0)[true_label].detach().cpu().item())

            if pred_label != true_label:
                # store the raw image for plotting (uint8 HWC)
                misclassified.append({
                    "idx": idx,
                    "img": cur_img,
                    "true": true_label,
                    "pred": pred_label,
                    "pred_prob": pred_prob,
                    "true_prob": true_prob,
                })

            # status + confusion breakdown
            if num_classes == 2:
                status = categorize(pred_label, true_label, idx, t_p, t_n, f_p, f_n)
            else:
                status = "Correct" if correct else "Incorrect"

            rows.append({
                "id": idx,
                "predicted class": class_name_map.get(pred_label, str(pred_label)),
                "true class": class_name_map.get(true_label, str(true_label)),
                "status": status
            })

    # Build dataframe once (fast)
    pred_df = pd.DataFrame(rows, columns=META_COL_NAMES)

    # Save predictions CSV (optional but usually useful)
    csv_path = output_folder / f"predictions-{opt.model_type}-{opt.set}-{opt.loading_epoch}.csv"
    pred_df.to_csv(csv_path, index=False)

    # --- Summary metrics ---
    accuracy = 100.0 * correct_counts / max(total_counts, 1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    f1 = f1_score(y_true, y_pred, average='macro') * 100.0

    # Confusion matrix plot
    plot_confusion_matrix(
        output_folder, cm,
        [class_name_map[i] for i in range(num_classes)],
        normalize=True,
        filename=f'confusion-matrix-{opt.model_type}-{opt.set}-{opt.loading_epoch}.png',
        title=f'Confusion Matrix\nOverall Accuracy: {accuracy:.6f}%\nMacro F1: {f1:.6f}%'
    )

    print(f'Accuracy on {total_counts} {opt.set} images: {accuracy:.6f}%')
    print(f'Macro F1: {f1:.6f}%')
    print(f'Saved CSV: {csv_path}')

    # Optional binary breakdown
    if num_classes == 2:
        infected_counts = int((labels_1d == 1).sum())
        clear_counts = int((labels_1d == 0).sum())
        print(f'Clear: {clear_counts} | Infected: {infected_counts}')

    if opt.save_misclassified:
        montage_path = output_folder / f"misclassified-{opt.model_type}-{opt.set}-{opt.loading_epoch}.png"
        save_misclassified_montage(
            misclassified=misclassified,
            class_name_map=class_name_map,
            out_path=montage_path,
            n=opt.n_misclassified,
            seed=2030
        )
