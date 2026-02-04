import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import math

import csv
from datetime import datetime

import torch
import torchvision.transforms as tvtrans

from utils import parse_model, load_model  # your utils file

# ---- Config
PATCH = 224


def remove_small_components(binmask, min_area=30):
    # binmask: 0/1 uint8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
    out = np.zeros_like(binmask)
    for i in range(1, num):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def leaf_mask(img, rel_th):
    """
        Fill holes in a binary image using cv2
        # Relative thresholding (0 < rth < 1)
        #
    Args:
        img:        A RGB format image with PIL format
        rel_th:     Relative threshold value
    Return:
        im_out:     Foreground mask with Numpy array format
    """
    img = np.asarray(img.convert("RGB"))  # ensure RGB

    wsize = 50
    wstep = 6

    res_offset = int(wsize / wstep)
    # center_offset = wsize / 2

    height, width, _ = img.shape
    aux_height = int((height / wstep) - res_offset)
    aux_width = int((width / wstep) - res_offset)

    # Output focal metric image
    fmat = np.ndarray(shape=(aux_height, aux_width), dtype=np.float64)

    coor_x = coor_y = 0

    # Get sharpness/focused of the image by calculating the standard deviation
    # Using sliding window
    for row in range(aux_height):
        for col in range(aux_width):
            kernel = img[coor_y: coor_y + wsize, coor_x: coor_x + wsize, :]
            _, stddev = cv2.meanStdDev(kernel)
            maxfm = stddev[0]
            for i in range(1, 3):
                maxfm = max(maxfm, stddev[i])
            fmat[row, col] = maxfm
            coor_x = coor_x + wstep
        coor_x = 0
        coor_y = coor_y + wstep

    _, max_val, _, _ = cv2.minMaxLoc(fmat)
    # 1st Threshold
    _, imbin = cv2.threshold(fmat, max_val * rel_th,
                             255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Calculate the mean between mask and thresholded mask
    fv = cv2.mean(fmat, imbin)
    th = fv[0]
    # 2nd Threshold
    _, imbin = cv2.threshold(fmat, th, 255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Bitwise not
    imbin1 = cv2.bitwise_not(imbin)
    imbin1 = imfill(imbin1)
    # Erode
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)

    h, w = imbin1.shape
    # Add margins for erosion
    for col in range(w):
        imbin1[0, col] = 0
        imbin1[h - 1, col] = 0
    for row in range(h):
        imbin1[row, 0] = 0
        imbin1[row, w - 1] = 0
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=4)
    imbin1 = imfill(imbin1)

    # Mask validation
    # The stats var is [num_of_connected_component, 5 (left, top, width, height, area)]
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(
        imbin1, connectivity=8, ltype=cv2.CV_16U)

    # No leaf pixels found
    if retval < 2:
        print('No leaf pixels found. No mask.')
        return None

    # Leaf area so small, skip
    big_l = 1
    for i in range(1, retval):
        if stats[i][4] > stats[big_l][4]:
            big_l = i
    if stats[big_l][4] < (h * w * 0.15):
        print('Leaf area so small! No mask.')
        return None

    coor_x = coor_y = 0
    # Delete other objects, keep the biggest
    for row in range(aux_height):
        for col in range(aux_width):
            imbin1[row, col] = 255 if labels[row, col] == big_l else 0

    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)
    imbin1 = cv2.resize(imbin1, dsize=(width, height), fx=0, fy=0,
                        interpolation=cv2.INTER_NEAREST)

    return imbin1


def write_csv(rows, out_csv_path):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

def imfill(inmask):
    """
        Do floodFill operation on the input mask
    """
    flood_mask = inmask.copy()
    h, w = flood_mask.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for col in range(w):
        if flood_mask[0, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, 0),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[h - 1, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, h - 1),
                          newVal=255, loDiff=10, upDiff=10)

    for row in range(h):
        if flood_mask[row, 0] == 0:
            cv2.floodFill(flood_mask, mask, (0, row),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[row, w - 1] == 0:
            cv2.floodFill(flood_mask, mask, (w - 1, row),
                          newVal=255, loDiff=10, upDiff=10)

    out_mask = inmask.copy()

    for row in range(h):
        for col in range(w):
            if flood_mask[row, col] == 0:
                out_mask[row, col] = 255

    return out_mask


def on_focus(imask, th=0.7):
    """
        Check image patches focused or not
    Args:
        imask:  Image patches' mask
    """
    mask_ratio = np.mean(imask)
    if mask_ratio > th:
        focused = True
    else:
        focused = False

    return focused


def patch_coords(H, W, patch=224, step=224):
    """
    Generate top-left coords that cover the full image via padding strategy.
    """
    ys = list(range(0, max(1, H - patch + 1), step))
    xs = list(range(0, max(1, W - patch + 1), step))
    # Ensure right/bottom edge coverage
    if ys[-1] != H - patch:
        ys.append(max(0, H - patch))
    if xs[-1] != W - patch:
        xs.append(max(0, W - patch))
    return xs, ys




def _pad_to_cover(H, W, patch, step):
    # smallest padded size so sliding window with stride covers edges
    ny = math.ceil((H - patch) / step) + 1
    nx = math.ceil((W - patch) / step) + 1
    Hpad = (ny - 1) * step + patch
    Wpad = (nx - 1) * step + patch
    pad_h = Hpad - H
    pad_w = Wpad - W
    return Hpad, Wpad, pad_h, pad_w


import contextlib
import torch
import torch.nn.functional as F
import numpy as np

PATCH = 224


def _pad_to_cover(H, W, patch, step):
    ny = math.ceil((H - patch) / step) + 1
    nx = math.ceil((W - patch) / step) + 1
    Hpad = (ny - 1) * step + patch
    Wpad = (nx - 1) * step + patch
    pad_h = Hpad - H
    pad_w = Wpad - W
    return Hpad, Wpad, pad_h, pad_w


@torch.inference_mode()
def infer_full_leaf_unfold_fold(
        model,
        device,
        img_pil,
        leaf_bin=None,  # optional (H,W) uint8 0/1
        step=224,
        outdim=2,
        prob_th=0.95,
        batch_size=64,
        means=(118 / 255, 165 / 255, 92 / 255),
        stds=(40 / 255, 35 / 255, 51 / 255),
        use_amp=True,
        min_area=200,
        rel_th=0.20,  # used only if we need to compute leaf mask here
):
    """
    Returns:
      pred_full_np: (H,W) uint8 0/1
      leaf_bin_np:  (H,W) uint8 0/1
    """
    assert step > 0
    assert model is not None

    rgb = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]

    # --- leaf mask (CPU) if not provided
    if leaf_bin is None:
        leaf = leaf_mask(img_pil, rel_th=rel_th)  # 0/255 or None
        if leaf is None:
            leaf_bin_np = np.zeros((H, W), dtype=np.uint8)
        else:
            leaf_bin_np = (leaf > 0).astype(np.uint8)
    else:
        leaf_bin_np = leaf_bin.astype(np.uint8)

    # If no leaf pixels, return empty prediction
    if leaf_bin_np.sum() == 0:
        pred_full_np = np.zeros((H, W), dtype=np.uint8)
        return pred_full_np, leaf_bin_np

    # --- Image to GPU once
    x = torch.from_numpy(rgb).to(device=device)
    x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    mean = torch.tensor(means, device=device).view(1, 3, 1, 1)
    std = torch.tensor(stds, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    # --- Pad so unfold covers full image
    Hpad, Wpad, pad_h, pad_w = _pad_to_cover(H, W, PATCH, step)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")  # (1,3,Hpad,Wpad)

    # --- Unfold patches: (1, 3*PATCH*PATCH, L)
    patches = F.unfold(x, kernel_size=PATCH, stride=step)
    L = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(L, 3, PATCH, PATCH)

    # --- Run model in GPU batches -> collect logits on GPU
    logits_chunks = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        for i in range(0, L, batch_size):
            xb = patches[i:i + batch_size]
            out = model(xb)["out"]  # (B,2,PATCH,PATCH)
            logits_chunks.append(out.float())  # accumulate fp32

    logits = torch.cat(logits_chunks, dim=0)  # (L,2,PATCH,PATCH)

    # --- Fold logits back (avg over overlaps)
    logits_flat = logits.reshape(L, 2 * PATCH * PATCH).transpose(0, 1).unsqueeze(0)
    logit_sum = F.fold(
        logits_flat, output_size=(Hpad, Wpad), kernel_size=PATCH, stride=step
    )  # (1,2,Hpad,Wpad)

    ones = torch.ones((L, 1, PATCH, PATCH), device=device, dtype=torch.float32)
    ones_flat = ones.reshape(L, PATCH * PATCH).transpose(0, 1).unsqueeze(0)
    count_map = F.fold(
        ones_flat, output_size=(Hpad, Wpad), kernel_size=PATCH, stride=step
    )
    count_map = torch.clamp(count_map, min=1.0)

    logit_avg = logit_sum / count_map  # (1,2,Hpad,Wpad)

    # --- Threshold using margin equivalent to softmax prob_th
    T = float(np.clip(prob_th, 1e-6, 1 - 1e-6))
    M = float(np.log(T / (1.0 - T)))

    margin = logit_avg[:, 1] - logit_avg[:, 0]  # (1,Hpad,Wpad)
    pred = (margin > M).to(torch.uint8)  # (1,Hpad,Wpad)

    # crop to original size
    pred = pred[:, :H, :W].squeeze(0)  # (H,W)

    # apply leaf mask (GPU)
    leaf_t = torch.from_numpy(leaf_bin_np).to(device=device, dtype=torch.uint8)
    pred = pred & leaf_t  # both are 0/1 uint8

    # CPU postprocess once
    pred_full_np = pred.cpu().numpy().astype(np.uint8)
    pred_full_np = remove_small_components(pred_full_np, min_area=min_area)

    return pred_full_np, leaf_bin_np


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="DeepLab")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--loading_epoch", type=int, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--outdim", type=int, default=2)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda_id", default="0")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--in_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--step", type=int, default=224, help="use 112 for overlap, 224 for no overlap")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rel_th", type=float, default=0.20)

    parser.add_argument("--prob_th", type=float, default=0.7, help="foreground probability threshold")
    parser.add_argument("--min_area", type=int, default=200, help="min component area to keep")
    parser.add_argument("--leaf_patch_min", type=float, default=0.05, help="skip patches with leaf fraction below this")
    parser.add_argument("--csv_name", type=str, default="inference_summary.csv")

    args = parser.parse_args()

    model_para = parse_model(args)
    model, device = load_model(model_para)

    in_folder = Path(args.in_folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    imgs = sorted([p for p in in_folder.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        raise RuntimeError(f"No images found in {in_folder}")

    run_time = datetime.now().isoformat(timespec="seconds")
    rows = []

    for p in imgs:
        img = Image.open(p).convert("RGB")

        pred_full, leaf_bin = infer_full_leaf_unfold_fold(
            model=model,
            device=device,
            img_pil=img,
            step=args.step,
            outdim=args.outdim,
            prob_th=args.prob_th,
            batch_size=args.batch_size,
            use_amp=True,
            min_area=args.min_area,
            rel_th=args.rel_th,  # only used if leaf_bin isn't provided
        )

        # Save output mask
        out_mask = (pred_full * 255).astype(np.uint8) if args.outdim == 2 else pred_full.astype(np.uint8)
        out_path = out_folder / f"{p.stem}_leaf_pred.png"
        cv2.imwrite(str(out_path), out_mask)

        # ---- Metrics
        pred_white = int(pred_full.sum())  # since mask is 0/1
        leaf_pixels = int(leaf_bin.sum())  # leaf_bin is 0/1
        total_pixels = int(pred_full.size)
        pct_in_leaf = (100.0 * pred_white / leaf_pixels) if leaf_pixels > 0 else 0.0
        pct_total = 100.0 * pred_white / total_pixels

        rows.append({
            "filename": p.name,
            "stem": p.stem,
            "run_time": run_time,

            "model_type": args.model_type,
            "model_timestamp": args.timestamp,
            "epoch": args.loading_epoch,

            "rel_th": args.rel_th,
            "prob_th": args.prob_th,
            "min_area": args.min_area,
            "step": args.step,
            "batch_size": args.batch_size,
            "leaf_patch_min": args.leaf_patch_min,

            "pred_white_pixels": pred_white,
            "leaf_pixels": leaf_pixels,
            "total_pixels": total_pixels,
            "pct_infected_of_leaf": round(pct_in_leaf, 4),
            "pct_infected_of_image": round(pct_total, 4),

            "mask_path": str(out_path),
        })

        print(
            f"Saved: {out_path} | infected_pixels={pred_white} | leaf_pixels={leaf_pixels} | pct_leaf={pct_in_leaf:.2f}%")

    # Write spreadsheet(s)
    csv_path = out_folder / args.csv_name
    write_csv(rows, csv_path)
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
