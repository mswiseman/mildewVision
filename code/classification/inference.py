import os
import h5py
import glob
import argparse
import numpy as np
import pandas as pd
import sys

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


import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def cache_dual_head_probs(images, labels_1d, model, test_transform, device, amp_enabled: bool):
    p_inf_all = np.zeros(len(images), dtype=np.float32)
    p_spor_all = np.full(len(images), np.nan, dtype=np.float32)  # nan if no spor head
    y_true = labels_1d.astype(np.int64, copy=False)

    model.eval()
    with torch.no_grad():
        for i in range(len(images)):
            cur_img = images[i]
            if isinstance(cur_img, np.ndarray) and cur_img.ndim == 2:
                cur_img = np.stack([cur_img] * 3, axis=-1)

            x = test_transform(cur_img).unsqueeze(0).to(device)

            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    logits = model(x)
            else:
                logits = model(x)

            probs = torch.sigmoid(logits).detach().cpu().numpy()  # (1,H)
            p_inf_all[i] = float(probs[0, 0])
            if probs.shape[1] >= 2:
                p_spor_all[i] = float(probs[0, 1])

    return y_true, p_inf_all, p_spor_all


def cache_softmax_probs(images, labels_1d, model, test_transform, device, amp_enabled: bool):
    """
    For non-dual-head softmax binary models:
      p_inf = P(class 1 = infected)
      p_spor = all NaN (unused)
    """
    p_inf_all = np.zeros(len(images), dtype=np.float32)
    p_spor_all = np.full(len(images), np.nan, dtype=np.float32)
    y_true = labels_1d.astype(np.int64, copy=False)

    model.eval()
    with torch.no_grad():
        for i in range(len(images)):
            cur_img = images[i]
            if isinstance(cur_img, np.ndarray) and cur_img.ndim == 2:
                cur_img = np.stack([cur_img] * 3, axis=-1)

            x = test_transform(cur_img).unsqueeze(0).to(device)

            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    logits = model(x)
            else:
                logits = model(x)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # (1,2)
            # infected prob assumed index 1
            p_inf_all[i] = float(probs[0, 1])

    return y_true, p_inf_all, p_spor_all


def predict_from_thresholds(p_inf, p_spor, down_th, up_th, spor_th, inf_gate,
                            dpi_gt5_prefer_spor=True):
    """
    Returns:
      y_pred (int array)
      discard_mask (bool array) where p_inf is in (down_th, up_th)
    """
    n = len(p_inf)
    y_pred = np.zeros(n, dtype=np.int64)
    discard = (p_inf > down_th) & (p_inf < up_th)

    # start with clear
    y_pred[p_inf <= down_th] = 0

    # candidate infected region
    infected_region = (p_inf >= up_th)
    y_pred[infected_region] = 1

    # spor call: only where spor head exists (not nan), p_inf passes gate, p_spor passes threshold
    has_spor = ~np.isnan(p_spor)
    spor_region = has_spor & (p_inf > inf_gate) & (p_spor >= spor_th)

    # apply precedence
    if dpi_gt5_prefer_spor:
        # spor overrides infected if both true
        y_pred[spor_region] = 2
    else:
        # infected wins if infected_region; spor only where not confidently infected
        y_pred[spor_region & ~infected_region] = 2

    # discard zone: keep label 0 placeholder (or whatever), but mark discard so caller can ignore
    y_pred[discard] = 0

    return y_pred, discard


def detect_softmax_outdim(model, sample_img_tensor, device, amp_enabled: bool) -> int:
    model.eval()
    with torch.no_grad():
        if amp_enabled:
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(sample_img_tensor.to(device))
        else:
            logits = model(sample_img_tensor.to(device))
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    return int(logits.shape[1])


def collapse_test_labels_if_needed(labels_1d: np.ndarray, model_outdim: int) -> np.ndarray:
    present = np.unique(labels_1d)
    test_num_classes = len(present)
    if test_num_classes == 3 and model_outdim == 2:
        # collapse class 2 into class 1 (infected)
        return np.where(labels_1d == 2, 1, labels_1d)
    return labels_1d


def score_predictions(y_true, y_pred, discard_mask=None,
                      ignore_discard=False, collapse_spor_to_infected=False):
    yt = y_true.copy()
    yp = y_pred.copy()

    if collapse_spor_to_infected:
        yt = np.where(yt == 2, 1, yt)
        yp = np.where(yp == 2, 1, yp)

    if ignore_discard and discard_mask is not None:
        keep = ~discard_mask
        yt = yt[keep]
        yp = yp[keep]

    # choose labels based on what’s present after collapse
    labels = sorted(np.unique(np.concatenate([yt, yp])))
    macro_f1 = f1_score(yt, yp, average="macro") if len(labels) > 1 else 0.0
    cm = confusion_matrix(yt, yp, labels=labels)

    return macro_f1, cm, labels


def grid_search_thresholds(
        y_true, p_inf, p_spor,
        down_grid, up_grid,
        spor_grid=None, gate_grid=None,
        dual_head=False,
        ignore_discard=True,
        collapse_spor_to_infected=False,
        dpi_gt5_prefer_spor=True,
        top_k=25,
        default_spor_th=0.8,
        default_inf_gate=0.3
):
    results = []

    for down_th in down_grid:
        for up_th in up_grid:
            if up_th <= down_th:
                continue

            # If NOT dual head: run once using default values
            if not dual_head:
                spor_values = [default_spor_th]
                gate_values = [default_inf_gate]
            else:
                # If dual head: loop over full grids
                spor_values = spor_grid
                gate_values = gate_grid

            for spor_th in spor_values:
                for inf_gate in gate_values:
                    y_pred, discard = predict_from_thresholds(
                        p_inf, p_spor,
                        down_th=down_th, up_th=up_th,
                        spor_th=spor_th, inf_gate=inf_gate,
                        dpi_gt5_prefer_spor=dpi_gt5_prefer_spor
                    )

                    macro_f1, cm, labels = score_predictions(
                        y_true, y_pred, discard_mask=discard,
                        ignore_discard=ignore_discard,
                        collapse_spor_to_infected=collapse_spor_to_infected
                    )

                    results.append(
                        (macro_f1, down_th, up_th, spor_th, inf_gate, cm, labels)
                    )

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def pred_img(img, model, dual_head: bool, use_autocast: bool = False):
    """
    Returns:
      - pred_label (int)
      - probs (torch.Tensor):
           * softmax mode: (1,2) probs for classes [clear, infected]
           * dual_head mode: (1,H) sigmoid probs for heads [infected,(spor)]
      - extra (dict): contains p_inf, p_spor (if dual_head), and optional "discard" flag
    """
    extra = {}

    if use_autocast:
        # caller should wrap autocast context; keep function simple
        logits = model(img)
    else:
        logits = model(img)

    if not dual_head:
        # old binary: logits (1,2)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        return pred, probs, extra

    # dual-head: logits (1,1) or (1,2)
    probs = torch.sigmoid(logits)
    p_inf = float(probs[0, 0].item())
    p_spor = float(probs[0, 1].item()) if probs.shape[1] >= 2 else None

    extra["p_inf"] = p_inf
    extra["p_spor"] = p_spor

    # label is decided OUTSIDE using thresholds (so inference loop can apply dpi logic if desired)
    # Here we just return a placeholder pred (0), actual pred computed in loop
    return 0, probs, extra


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
    parser.add_argument('--collapse_spor_to_infected', action='store_true')

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

    parser.add_argument('--dual_head', action='store_true',
                        help='If set: interpret model outputs as sigmoid heads (infected[, spor]). '
                             'If not set: interpret outputs as softmax classes (clear vs infected).')
    parser.add_argument('--grid_search', action='store_true', help='perform a grid search on threshold values')
    parser.add_argument('--up_threshold', type=float, default=0.7,
                        help='infected threshold for p_inf when dual_head')
    parser.add_argument('--down_threshold', type=float, default=0.1,
                        help='clear threshold for p_inf when dual_head')
    parser.add_argument('--spor_th', type=float, default=None,
                        help='sporulation threshold for p_spor (dual_head). Defaults to up_threshold.')
    parser.add_argument('--inf_gate', type=float, default=None,
                        help='minimum p_inf required to allow sporulation call (dual_head). Defaults to down_threshold.')
    parser.add_argument('--ignore_discard', action='store_true',
                        help='If set (dual_head only), discard samples are excluded from metrics.')

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
        test_filepath = 'testv2.hdf5'

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
    amp_enabled = bool(opt.amp) and (device.type == "cuda")

    present_before = sorted(np.unique(labels_1d).tolist())
    test_num_classes = len(present_before)

    # build one sample tensor to probe model output dim
    sample_img = images[0]
    if isinstance(sample_img, np.ndarray) and sample_img.ndim == 2:
        sample_img = np.stack([sample_img] * 3, axis=-1)
    sample_x = test_transform(sample_img).unsqueeze(0)

    model_outdim = detect_softmax_outdim(model, sample_x, device, amp_enabled=amp_enabled)

    if not opt.dual_head and model_outdim != test_num_classes:
        print(f"[Class check] test labels present: {present_before} (n={test_num_classes})")
        print(f"[Class check] model softmax outdim: {model_outdim}")


        labels_1d = collapse_test_labels_if_needed(labels_1d, model_outdim)

        present_after = sorted(np.unique(labels_1d).tolist())
        if present_after != present_before:
            print(f"[Class check] Collapsed test labels: {present_before} -> {present_after} (collapsed 2 into 1)")

    # --- Output records ---
    META_COL_NAMES = ['id', 'predicted class', 'true class', 'status']
    rows = []  # collect dicts/rows and build DataFrame once

    if opt.dual_head:
        num_classes = 3
    else:
        num_classes = model_outdim

    class_name_map = _default_class_name_map(num_classes)

    # Metrics accumulators
    y_true: list[int] = []
    y_pred: list[int] = []

    # Binary-only confusion breakdown lists (only meaningful if num_classes == 2)
    f_n, f_p, t_n, t_p = [], [], [], []
    misclassified = []  # list of dicts: {idx, img, true, pred}

    if opt.grid_search:
        if opt.dual_head:
            print("Caching dual-head probabilities once...")
            y_true, p_inf, p_spor = cache_dual_head_probs(
                images=images,
                labels_1d=labels_1d,
                model=model,
                test_transform=test_transform,
                device=device,
                amp_enabled=amp_enabled
            )
        else:
            print("Caching softmax probabilities once...")
            y_true, p_inf, p_spor = cache_softmax_probs(
                images=images,
                labels_1d=labels_1d,
                model=model,
                test_transform=test_transform,
                device=device,
                amp_enabled=amp_enabled
            )

        print("Running grid search...")
        down_grid = np.round(np.linspace(0.05, 0.35, 5), 2)
        up_grid = np.round(np.linspace(0.50, 0.95, 8), 2)

        # Only needed/used when dual_head=True; safe to define always
        spor_grid = np.round(np.linspace(0.50, 0.90, 9), 2)
        gate_grid = np.round(np.linspace(0.20, 0.60, 5), 2)

        top = grid_search_thresholds(
            y_true=y_true,
            p_inf=p_inf,
            p_spor=p_spor,  # NaNs if not dual_head
            down_grid=down_grid,
            up_grid=up_grid,
            spor_grid=spor_grid,
            gate_grid=gate_grid,
            dual_head=opt.dual_head,
            ignore_discard=opt.ignore_discard,
            collapse_spor_to_infected=opt.collapse_spor_to_infected,
            dpi_gt5_prefer_spor=True,
            top_k=20,
            default_spor_th=0.8,
            default_inf_gate=0.3
        )

        print("\n===== TOP 5 THRESHOLD SETTINGS =====")

        for rank, res in enumerate(top[:5], start=1):
            macro_f1, down_th, up_th, spor_th, inf_gate, cm, labels = res

            print(f"\n--- Rank {rank} ---")
            print(f"Macro F1: {macro_f1:.4f}")
            print(
                f"down_th={down_th}, up_th={up_th}"
                + (f", spor_th={spor_th}, inf_gate={inf_gate}" if opt.dual_head else "")
            )
            print("labels:", labels)
            print("confusion matrix:\n", cm)

        sys.exit(0)

    print("INFERENCE START")

    total_counts = int(images.shape[0])  # stays constant

    # thresholds (compute once)
    up_th = float(opt.up_threshold)
    down_th = float(opt.down_threshold)
    down_th = float(opt.down_threshold)
    spor_th = float(opt.spor_th) if opt.spor_th is not None else up_th
    inf_gate = float(opt.inf_gate) if opt.inf_gate is not None else down_th

    # Reset accumulators (ensure empty before loop)
    y_true = []
    y_pred = []
    rows = []
    misclassified = []
    correct_counts = 0

    with torch.no_grad():
        for idx in range(total_counts):
            cur_img = images[idx]

            # grayscale -> fake RGB if needed
            if isinstance(cur_img, np.ndarray) and cur_img.ndim == 2:
                cur_img = np.stack([cur_img] * 3, axis=-1)

            preproc_img = test_transform(cur_img).unsqueeze(0).to(device)
            true_label = int(labels_1d[idx])

            # forward
            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    _, prob_t, extra = pred_img(preproc_img, model, dual_head=opt.dual_head)
            else:
                _, prob_t, extra = pred_img(preproc_img, model, dual_head=opt.dual_head)

            discard = False
            pred_probs_for_reporting = None
            true_prob_for_reporting = None

            if not opt.dual_head:
                # softmax binary -> predicted class 0/1
                pred_label = int(torch.argmax(prob_t, dim=1).item())
                pred_probs_for_reporting = float(prob_t[0, pred_label].detach().cpu().item())
                true_prob_for_reporting = float(prob_t[0, true_label].detach().cpu().item())

            else:
                # sigmoid heads -> thresholds -> map to 0/1/2
                p_inf = float(extra["p_inf"])
                p_spor = float(extra["p_spor"]) if extra.get("p_spor") is not None else None

                if p_inf <= down_th:
                    pred_label = 0  # clear
                elif p_inf >= up_th:
                    pred_label = 2 if (p_spor is not None and p_inf > inf_gate and p_spor >= spor_th) else 1
                else:
                    discard = True
                    pred_label = 0  # placeholder label for “discard band”

                if opt.ignore_discard and discard:
                    continue

                # reporting probs for montage
                if pred_label == 0:
                    pred_probs_for_reporting = 1.0 - p_inf
                elif pred_label == 1:
                    pred_probs_for_reporting = p_inf
                else:
                    pred_probs_for_reporting = p_spor if p_spor is not None else float("nan")

                if true_label == 0:
                    true_prob_for_reporting = 1.0 - p_inf
                elif true_label == 1:
                    true_prob_for_reporting = p_inf
                else:
                    true_prob_for_reporting = p_spor if p_spor is not None else float("nan")

            # collapse (if requested) BEFORE metrics/rows
            if opt.collapse_spor_to_infected:
                if pred_label == 2:
                    pred_label = 1
                if true_label == 2:
                    true_label = 1

            # bookkeeping
            y_true.append(true_label)
            y_pred.append(pred_label)

            correct = (pred_label == true_label)
            correct_counts += int(correct)

            # misclassified
            if pred_label != true_label:
                misclassified.append({
                    "idx": idx,
                    "img": cur_img,
                    "true": true_label,
                    "pred": pred_label,
                    "pred_prob": pred_probs_for_reporting,
                    "true_prob": true_prob_for_reporting,
                })

            # status + rows
            if (not opt.dual_head) and num_classes == 2:
                status = categorize(pred_label, true_label, idx, t_p, t_n, f_p, f_n)
            else:
                status = "Correct" if correct else "Incorrect"

            rows.append({
                "id": idx,
                "predicted class": class_name_map.get(pred_label, str(pred_label)),
                "true class": class_name_map.get(true_label, str(true_label)),
                "status": status
            })

            # optional: progress print every so often
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{total_counts}")

    # Build dataframe once (fast)
    pred_df = pd.DataFrame(rows, columns=META_COL_NAMES)

    # Save predictions CSV (optional but usually useful)
    csv_path = output_folder / f"predictions-{opt.model_type}-{opt.set}-{opt.loading_epoch}.csv"
    pred_df.to_csv(csv_path, index=False)

    # --- Summary metrics ---
    evaluated_counts = len(y_true)
    accuracy = 100.0 * correct_counts / max(evaluated_counts, 1)

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

    print(f'Accuracy on {evaluated_counts} {opt.set} images: {accuracy:.6f}%')
    print(f'(Discarded: {total_counts - evaluated_counts})')
    print(f'Macro F1: {f1:.6f}%')
    print(f'Saved CSV: {csv_path}')

    # Optional binary breakdown
    if not opt.dual_head:
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
            seed=2031
        )
