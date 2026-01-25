# Standard library
import argparse
import gc
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

# Local project modules
from analyzer_config import IMG_HEIGHT, IMG_WIDTH
from metric import patch_sr, pixel_sr1
from classification.inference import pred_img
from classification.utils import (
    load_model,
    parse_model,
    printArgs,
    set_logging,
    timeSince,
    adaptive_threshold
)
from analysis.leaf_mask import leaf_mask, on_focus
from visualization.viz_helper import (
    get_first_conv_layer,
    get_last_conv_layer,
    normalize_image_attr
)
from sanity_check.utils import get_saliency_methods, get_saliency_masks

np.random.seed(2020)

""" Usage
Analyze the full-size leaf disc images and calculate the severity rate
Given a date, do analysis on all the data collected in that date
"""

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--model_type', default='VGG', help='model used for training')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model parameters')
parser.add_argument('--loading_epoch', type=int, required=True, help='xth model loaded for inference')
parser.add_argument('--timestamp', required=True, help='model timestamp')
parser.add_argument('--outdim', type=int, default=1,
                    help='number of model outputs: 1=binary infected head, 2=dual-head (infected + sporulating)')
parser.add_argument('--dual_head', action='store_true',
                    help='Use a dual-head model (infected + sporulating). If not set, inferred from outdim')
parser.add_argument('--model_path', type=str, required=True, help='root path to the model')
parser.add_argument('--step_size', type=int, default=224, help='step size of sliding window')
parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361],
                    help='List of means for each channel')
parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192],
                    help='List of standard deviations for each channel')
parser.add_argument(
    '--target_class', type=int, default=1,
    help='saliency target head (0=infected, 1=sporulating for outdim=2; use 0 for outdim=1)')
parser.add_argument('--contam_control', action='store_true', help='use contamination control conditional logic')
parser.add_argument('--spor_th', type=float, default=None,
                    help='sporulation threshold for dual-head (defaults to up_threshold if not set)')
parser.add_argument('--inf_gate', type=float, default=None,
                    help='minimum infected prob required to allow sporulation call (defaults to down_threshold)')
parser.add_argument('--pm', type=str, help='pm isolate for metadata')

# CPU/GPU/MSP parameters
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id', default="0", help='specify cuda id')

# Output parameters
parser.add_argument('--sal_threshold', type=float, default=0.5, help='threshold for saliency map')

# Data analysis parameters
parser.add_argument('--up_threshold', type=float, default=0.8, help='upper threshold for severity ratio')
parser.add_argument('--down_threshold', type=float, default=0.2, help='lower threshold for severity ratio')
parser.add_argument('--dataset_path', type=str, required=True, help='root path to the data')
parser.add_argument('--img_folder', type=str, default="2-5-2023_6dpi", help='directory of images')
parser.add_argument('--platform', type=str, default='BlackBird', help='robot platform (Pmbot or BlackBird)')
parser.add_argument('--threshold', nargs='+', help='thresholding value for pixel sr')
parser.add_argument('--log', type=str, default='../../results/logs/random.log', help='log file path')
parser.add_argument('--dpi', type=int, required=True, help='inoculation date')
parser.add_argument('--group', type=str, default='baseline', help='exp group')
parser.add_argument('--trays', nargs='+', help='trays')

# filter out routine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# saliency mapping flags
parser.add_argument('--sal_gradcam', action='store_true')
parser.add_argument('--sal_gradient', action='store_true')
parser.add_argument('--sal_smoothgrad', action='store_true')
parser.add_argument('--sal_deeplift', action='store_true')
parser.add_argument('--sal_thresh_method', type=str, default='percentile',
                    choices=['percentile', 'fixed'],
                    help='How to compute saliency threshold per image/method')
parser.add_argument('--sal_thresh_p', type=float, default=95.0,
                    help='Percentile used when method=percentile')

opt = parser.parse_args()

# set device
if opt.cuda and torch.cuda.is_available():
    device = torch.device(f"cuda:{opt.cuda_id}")
elif opt.mps and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# set logging options
logger = set_logging(Path(str(opt.log)), 20)
logger.info(os.path.basename(__file__))
printArgs(logger, vars(opt))

# set paths
ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test.hdf5',
}
image_timestamp = opt.img_folder
model_timestamp = opt.timestamp
model_type = opt.model_type

outdim = opt.outdim
# Dual-head flag (infected + sporulating)
dual_head = bool(getattr(opt, "dual_head", False))
dataset_path = Path(opt.dataset_path) / image_timestamp
mask_path = Path(opt.dataset_path) / f'{image_timestamp}_masking'
model_string = model_type + '_upth' + str(opt.up_threshold) + '_downth' + str(
    opt.down_threshold) + '_' + opt.timestamp
output_folder = Path(opt.dataset_path).parents[0] / 'Results' / model_string / image_timestamp

if dual_head and outdim < 2:
    raise ValueError("dual_head=True but outdim < 2. Set --outdim 2.")

if (not dual_head) and outdim >= 2:
    print("WARNING: outdim>=2 but dual_head flag not set. If you're using a dual-head model, please set --dual_head.")

# Threshold for severity ratio
down_th = opt.down_threshold  # below this will be classified as healthy
up_th = opt.up_threshold  # above this will be classified as infected or conidiophores
pixel_th = opt.threshold if opt.threshold else []
spor_th = opt.spor_th if opt.spor_th is not None else opt.up_threshold
inf_gate = opt.inf_gate if opt.inf_gate is not None else opt.down_threshold
overlay_thresh_fixed = float(opt.sal_threshold)
if not pixel_th:
    pixel_th = [overlay_thresh_fixed]

rel_th = 0.2  # relative threshold leaf mask
target_class = int(opt.target_class) if opt.target_class != 'None' else None
step_size = opt.step_size

# Model
model_para = parse_model(opt)
model, device = load_model(model_para)
model.eval()
last_conv_layer = get_last_conv_layer(model)
first_conv_layer = get_first_conv_layer(model)

means = opt.means
stds = opt.stds

# Preprocessing
if opt.model_type == 'Inception3':
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 299
if opt.dpi > 5:
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Lambda(lambda img: tvtrans.functional.adjust_brightness(img, 0.75)),  # improve conidiophore detection
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 224
else:
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 224

# Captum
use_saliency = any([opt.sal_gradcam, opt.sal_gradient, opt.sal_smoothgrad, opt.sal_deeplift])
if use_saliency:
    saliency_methods = get_saliency_methods(model,
                                            last_conv_layer=last_conv_layer,
                                            first_conv_layer=first_conv_layer,
                                            ref_dataset_path=ref_dataset_path,
                                            image_width=image_width,
                                            transform=preprocess,
                                            device=device,
                                            partial=True,
                                            explanation_map=False,
                                            gradcam=opt.sal_gradcam,
                                            gradient=opt.sal_gradient,
                                            smooth_grad=opt.sal_smoothgrad,
                                            deeplift=opt.sal_deeplift)
else:
    saliency_methods = {}

# Write severity ratio as CSV files
key = [f'{x}_sr2' for x in saliency_methods.keys()]


META_COL_NAMES = ['timestamp', 'time_elapsed', 'model_type', 'model_timestamp', 'classes', 'imaging_date', 'tray',
                  'filename', 'conserved_identifier', 'USDA_number', 'CHUM_number_if_from_NCGR', 'other_name', 'PM',
                  'infected_threshold', 'healthy_threshold', 'sal_threshold'] + \
                 (['inf_gate', 'spor_th'] if dual_head else []) + \
                 ['leaf_mask_th', 'clear_patches', 'hyphal_patches'] + \
                 (['conidiophore_patches', 'sporulating_pct'] if dual_head else []) + \
                 ['discarded_patches', 'severity_rate_patch'] + key

# List all trays
tray = opt.trays
PM = opt.pm

# threshold = 0.7  # threshold for saliency map

# Time
total_time = 0
total_time_2 = 0
format_ = 'png'

# Loop trays
for tray_id in tray:
    dataset_tray_path = dataset_path / Path(tray_id)
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path) if x.endswith('.png')]

    severity_rate_df_list = []
    for th in pixel_th:
        severity_rate_df_list.append(pd.DataFrame(columns=META_COL_NAMES))

    # Loop leaf disk images
    for leaf_disk_image_filename in leaf_disk_image_filenames:
        img_filepath = dataset_tray_path / leaf_disk_image_filename

        # Timer
        start_time = time.time()

        # Get current date and time
        now = datetime.now()

        # Format as a string
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_time_str_filename = now.strftime("%Y%m%d_%H%M%S")

        logger.info('-------------------------------------------')
        logger.info('Processing {} {} {}'.format(image_timestamp, tray_id, leaf_disk_image_filename))

        # Get info of resized image subim_x: number of patches one row
        img = Image.open(img_filepath)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_arr = np.asarray(img)
        width, height = img.size

        subim_x = (width - IMG_WIDTH) // step_size + 1
        subim_y = (height - IMG_HEIGHT) // step_size + 1
        subim_height = (subim_y - 1) * step_size + IMG_HEIGHT
        subim_width = (subim_x - 1) * step_size + IMG_WIDTH
        sub_img = img.crop((0, 0, subim_width, subim_height))
        sub_img_arr = np.asarray(sub_img)

        imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

        # Masking
        imask = leaf_mask(img, rel_th=rel_th)
        if imask is None:
            logger.info('Image: {}\tmasking ERROR'.format(imagename_text))
            continue
        imask = imask.astype('uint8') / 255

        t1 = time.time()
        logger.info('Finished loading mask: {}'.format(timeSince(start_time)))

        # Set variables to zero before looping patches
        patch_idx = coor_x = coor_y = 0
        infected_patch = conidiophore_patch = clear_patch = discard_patch = lost_focus_patch = total_patch = 0
        infected_pixel = conidiophore_pixel = clear_pixel = discard_pixel = lost_focus_pixel = total_pixel = 0

        # Counter of each pixel
        counting_map = np.zeros(shape=(height, width))
        prob_attrs1 = np.zeros(
            shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=float)
        if dual_head:
            prob_attrs2 = np.zeros(
                shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=float)

        saliency_attrs = {}
        for saliency_method_key in saliency_methods.keys():
            saliency_attrs[saliency_method_key] = np.zeros(
                shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=float)

        f = os.path.splitext(leaf_disk_image_filename)[0]

        # Crop
        for _ in range(subim_y):
            for _ in range(subim_x):
                subim_mask = imask[coor_y: coor_y + IMG_HEIGHT, coor_x: coor_x + IMG_WIDTH]
                if not on_focus(subim_mask):
                    # Set lost focused patches' pixel values as -inf
                    lost_focus_patch += 1
                    prob_attrs1[patch_idx] = -np.inf
                    if dual_head:
                        prob_attrs2[patch_idx] = -np.inf

                else:
                    # Cropping
                    box = (coor_x, coor_y, coor_x + IMG_WIDTH, coor_y + IMG_HEIGHT)
                    subim = img.crop(box).resize((image_width, image_height))
                    subim_arr = np.asarray(subim)

                    # Preprocess
                    input_img = preprocess(subim_arr).unsqueeze(0).to(device).requires_grad_(True)

                    # Forward pass -> probabilities (kept concordant with plot_sal_map_leaf.py)
                    logits = model(input_img)

                    if dual_head:
                        # Dual-head (multi-label): (1,2) logits -> sigmoid per head
                        probs = torch.sigmoid(logits).detach()
                        p_inf = float(probs[0, 0].cpu().item())
                        p_spor = float(probs[0, 1].cpu().item())
                    else:
                        # Binary: support BOTH conventions
                        #   (1,2) logits -> softmax, take P(class=infected)
                        #   (1,1) logit  -> sigmoid
                        if logits.ndim == 2 and logits.shape[1] == 2:
                            prob = torch.softmax(logits, dim=1).detach()
                            p_inf = float(prob[0, 1].cpu().item())
                        else:
                            prob = torch.sigmoid(logits).detach()
                            p_inf = float(prob[0, 0].cpu().item())
                        p_spor = None

                    # Store probs for reconstruction
                    prob_attrs1[patch_idx] = p_inf
                    if dual_head:
                        prob_attrs2[patch_idx] = p_spor

                    # Decide patch label + saliency head
                    patch_label = "discard"
                    target_head = 0

                    if not dual_head:
                        # binary: clear / infected / discard band
                        if p_inf >= up_th:
                            patch_label = "infected"
                            infected_patch += 1
                            target_head = 0
                        elif p_inf <= down_th:
                            patch_label = "clear"
                            clear_patch += 1
                        else:
                            discard_patch += 1
                    else:
                        # dual-head: clear / infected / sporulating / discard band

                        # (optional but strongly recommended) safety assert:
                        if p_spor is None:
                            raise RuntimeError("dual_head=True but p_spor is None. Check outdim/model outputs.")

                        # dpi-specific precedence (keep your original behavior)
                        if opt.dpi >= 5:
                            # spor first (gated), then infected
                            if (p_inf > inf_gate) and (p_spor >= spor_th):
                                patch_label = "spor"
                                conidiophore_patch += 1
                                target_head = 1
                            elif p_inf >= up_th:
                                patch_label = "infected"
                                infected_patch += 1
                                target_head = 0
                            elif p_inf <= down_th:
                                patch_label = "clear"
                                clear_patch += 1
                            else:
                                discard_patch += 1
                        else:
                            # infected first, then spor (gated)
                            if p_inf >= up_th:
                                patch_label = "infected"
                                infected_patch += 1
                                target_head = 0
                            elif (p_inf > inf_gate) and (p_spor >= spor_th):
                                patch_label = "spor"
                                conidiophore_patch += 1
                                target_head = 1
                            elif p_inf <= down_th:
                                patch_label = "clear"
                                clear_patch += 1
                            else:
                                discard_patch += 1

                    # --- Saliency (only if any saliency method is enabled) ---
                    if saliency_methods and patch_label in ("infected", "spor"):
                        output_masks = get_saliency_masks(
                            saliency_methods,
                            input_img,
                            target_head,
                            relu_attributions=True
                        )

                        # Normalize saliency maps
                        abs_norm, _, _ = normalize_image_attr(subim_arr, output_masks, hist=False)
                        abs_norm.pop('Original', None)

                        # Resize and store saliency maps
                        for key, val in abs_norm.items():
                            val_t = torch.as_tensor(val).unsqueeze(0).unsqueeze(0)
                            if image_height != IMG_HEIGHT:
                                val_t = F.interpolate(val_t, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')
                            saliency_attrs[key][patch_idx] = val_t[0, 0].cpu().numpy()

                    input_img.requires_grad_(False)
                # Update pixel counter each loop to avoid ZeroDivisionError
                counting_map[coor_y: coor_y + IMG_HEIGHT,
                coor_x: coor_x + IMG_WIDTH] += 1
                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        counting_map[counting_map == 0] = 1

        logger.info('Finished crop and inference: {}'.format(
            timeSince(start_time)))

        # Reconstruction
        prob_heatmap1 = np.zeros(
            shape=(height, width), dtype=float)
        if dual_head:
            prob_heatmap2 = np.zeros(
                shape=(height, width), dtype=float)
        saliency_heatmaps = {}
        for key in saliency_methods.keys():
            saliency_heatmaps[key] = np.zeros(
                shape=(height, width), dtype=float)

        patch_idx = coor_x = coor_y = 0
        for _ in range(subim_y):
            for _ in range(subim_x):
                prob_heatmap1[coor_y: coor_y + IMG_HEIGHT,
                coor_x: coor_x + IMG_WIDTH] += prob_attrs1[patch_idx]
                if dual_head:
                    prob_heatmap2[coor_y: coor_y + IMG_HEIGHT,
                    coor_x: coor_x + IMG_WIDTH] += prob_attrs2[patch_idx]

                for key in saliency_methods.keys():
                    saliency_heatmaps[key][coor_y: coor_y + IMG_HEIGHT,
                    coor_x: coor_x + IMG_WIDTH] += saliency_attrs[key][patch_idx]

                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        # Divide by counting_map
        prob_heatmap1 = prob_heatmap1 / counting_map
        if dual_head:
            prob_heatmap2 = prob_heatmap2 / counting_map

        for key, val in saliency_heatmaps.items():
            saliency_heatmaps[key] = val / counting_map
            # print("key: ", key, "val: ", val)

        # Severity rate calculation
        patch_info = {'infected_patch': infected_patch, 'conidiophore_patch': conidiophore_patch,
                      'clear_patch': clear_patch, 'discard_patch': discard_patch, 'lost_focus_patch': lost_focus_patch}
        heatmap_info = saliency_heatmaps.copy()
        heatmap_info['prob_heatmap1'] = prob_heatmap1

        # -------------------------------
        # Adaptive thresholds (once per image)
        # -------------------------------
        th_method = getattr(opt, 'sal_thresh_method', 'percentile')
        th_p = float(getattr(opt, 'sal_thresh_p', 95.0))
        adaptive_th = {
            k: (adaptive_threshold(v, mask=imask, method=th_method, p=th_p)
                if th_method != 'fixed' else overlay_thresh_fixed)
            for k, v in saliency_heatmaps.items()
        }

        # If user didn’t provide pixel thresholds, pick one from saliency adaptively
        if not pixel_th and adaptive_th:
            driver_key = 'GradCAM' if 'GradCAM' in adaptive_th else next(iter(adaptive_th))
            pixel_th = [float(adaptive_th[driver_key])]

        threshold_info = {
            'patch_down_th': down_th,
            'patch_up_th': up_th,
            'pixel_th': [float(x) for x in pixel_th] if pixel_th else [overlay_thresh_fixed],
        }

        if dual_head:
            heatmap_info['prob_heatmap2'] = prob_heatmap2

        # Calculate severity rate
        # print(f"Value of outdim: {outdim}")
        if dual_head:
            severity_rate_patch, pixels_patch = patch_sr.metric_two_class(
                patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, pixels_1 = pixel_sr1.metric(
                patch_info.copy(), heatmap_info.copy(), threshold_info.copy(), outdim)
        else:
            severity_rate_patch, pixels_patch = patch_sr.metric(
                patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, pixels_1 = pixel_sr1.metric(
                patch_info.copy(), heatmap_info.copy(), threshold_info.copy(), outdim)

        infected_patch = patch_info['infected_patch']
        conidiophore_patch = patch_info['conidiophore_patch']
        clear_patch = patch_info['clear_patch']
        discard_patches = discard_patch
        discarded_patches = patch_info['discard_patch']

        if opt.contam_control and opt.dpi > 6 and dual_head and conidiophore_patch < 2 and infected_patch > 10:
            infected_patch = "NA"
            conidiophore_patch = "NA"  # this is to catch contamination

        # % sporulating (only if outdim=3). Respect "NA" from contamination control and avoid divide-by-zero.
        sporulating_pct = None
        if dual_head:
            if isinstance(infected_patch, str) or isinstance(conidiophore_patch, str):
                sporulating_pct = "NA"
            else:
                total_patches = clear_patch + infected_patch + conidiophore_patch
                sporulating_pct = (conidiophore_patch / total_patches * 100.0) if total_patches > 0 else np.nan

        for i, th in enumerate(pixel_th):
            # Extract conserved_identifier from imagename_text
            conserved_identifier = imagename_text[4:].split('_')[0]

            record_data = [
                date_time_str,  # timestamp
                timeSince(start_time),  # time_elapsed
                model_type,  # model_type
                model_timestamp,  # model_timestamp
                outdim,  # classes
                image_timestamp,  # imaging_date
                tray_id,  # tray
                imagename_text,  # filename
                conserved_identifier,  # conserved_identifier
                '',  # USDA_number
                '',  # CHUM_number_if_from_NCGR
                '',  # other_name
                PM,  # PM

                up_th,  # infected_threshold
                down_th,  # healthy_threshold
                float(th),  # sal_threshold
            ]

            if dual_head:
                record_data += [inf_gate, spor_th]  # inf_gate, spor_th

            record_data += [
                rel_th,  # leaf_mask_th
                clear_patch,  # clear_patches
                infected_patch,  # hyphal_patches
            ]

            if dual_head:
                record_data += [
                    conidiophore_patch,  # conidiophore_patches
                    sporulating_pct  # sporulating_pct
                ]

            record_data += [
                               discard_patches,  # discarded_patches
                               severity_rate_patch  # severity_rate_patch
                           ] + list(severity_rates_pixel[float(th)].values())

            # sanity check
            if len(record_data) != len(META_COL_NAMES):
                raise ValueError(f"Row/column mismatch: {len(record_data)} values vs {len(META_COL_NAMES)} columns")

            record_df = pd.DataFrame([record_data], columns=META_COL_NAMES)
            severity_rate_df_list[i].loc[len(severity_rate_df_list[i])] = record_data

        total_time = total_time + time.time() - start_time
        total_time_2 = total_time_2 + time.time() - t1
        print(severity_rate_df_list[0])
        logger.info('Analysis finished: {}'.format(timeSince(start_time)))
        logger.info('-------------------------------------------')

    # mean_row = severity_rate_df_list[i].mean(numeric_only=True, axis=0)
    # std_row = severity_rate_df_list[i].std(numeric_only=True, axis=0)
    # min_row = severity_rate_df_list[i].min(numeric_only=True, axis=0)
    # max_row = severity_rate_df_list[i].max(numeric_only=True, axis=0)

    # severity_rate_df_list[i] = pd.concat([severity_rate_df_list[i], mean_row], ignore_index=True)
    # severity_rate_df_list[i] = pd.concat([severity_rate_df_list[i], std_row], ignore_index=True)
    # severity_rate_df_list[i] = pd.concat([severity_rate_df_list[i], min_row], ignore_index=True)
    # severity_rate_df_list[i] = pd.concat([severity_rate_df_list[i], max_row], ignore_index=True)

    for i, th in enumerate(pixel_th):
        output_csv_folder_th = output_folder / f'th_{th}'
        os.makedirs(output_csv_folder_th, exist_ok=True)
        out_path = output_csv_folder_th / f'severity_rate_tray_{tray_id}_u{up_th}_d{down_th}_ig{inf_gate}_sp{spor_th}_{date_time_str_filename}.csv'
        severity_rate_df_list[i].to_csv(out_path, index=False)
        logger.info('Saved %s', out_path)

    # Explicitly delete objects to free memory
    del img, img_arr, sub_img, sub_img_arr, prob_heatmap1, saliency_heatmaps

    if dual_head:
        del prob_heatmap2

    # Call garbage collector
    gc.collect()
