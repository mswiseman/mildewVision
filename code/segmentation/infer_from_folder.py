import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as tvtrans

# You already have these in your project
from utils import load_model, parse_model


def infer_folder(
        model,
        device,
        image_folder: Path,
        out_mask_folder: Path,
        means,
        stds,
        exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
        save_probs=False,  # optional: save foreground probability too
):
    out_mask_folder.mkdir(parents=True, exist_ok=True)

    transform = tvtrans.Compose([
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])

    image_paths = []
    for ext in exts:
        image_paths += list(image_folder.glob(f"*{ext}"))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise RuntimeError(f"No images found in {image_folder}")

    model.eval()
    print(f"Running inference on {len(image_paths)} patches...")

    with torch.no_grad():
        for p in image_paths:
            # Read image (RGB)
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable image: {p}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            x = transform(rgb).unsqueeze(0).to(device)  # (1,3,H,W)
            preds = model(x)

            logits = preds["out"]  # (1,2,H,W)
            prob_fg = torch.softmax(logits, dim=1)[:, 1]  # (1,H,W)

            mask = (prob_fg[0] > 0.6).byte().cpu().numpy()  # (H,W) uint8 0/1
            mask_255 = (mask * 255).astype(np.uint8)  # (H,W) uint8 0/255

            out_path = out_mask_folder / f"{p.stem}_pred.png"
            cv2.imwrite(str(out_path), mask_255)

            # Optional: save foreground probability map (for thresholding later)
            if save_probs and logits.shape[1] == 2:
                probs = prob_fg[0].cpu().numpy()  # (H,W)
                prob_path = out_mask_folder / f"{p.stem}_prob.png"
                cv2.imwrite(str(prob_path), (probs * 255).astype(np.uint8))

    print(f"Done. Masks saved to: {out_mask_folder}")


if __name__ == "__main__":
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

    parser.add_argument("--patch_folder", type=str, required=True)
    parser.add_argument("--out_mask_folder", type=str, required=True)
    parser.add_argument("--save_probs", action="store_true")

    opt = parser.parse_args()

    model_para = parse_model(opt)
    model, device = load_model(model_para)

    means = [118. / 255., 165. / 255., 92. / 255.]
    stds = [40. / 255., 35. / 255., 51. / 255.]

    infer_folder(
        model=model,
        device=device,
        image_folder=Path(opt.patch_folder),
        out_mask_folder=Path(opt.out_mask_folder),
        means=means,
        stds=stds,
        save_probs=opt.save_probs
    )
