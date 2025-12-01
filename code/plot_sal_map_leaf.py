# Standard library
import argparse
import gc
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

# Local imports
from analyzer_config import IMG_HEIGHT, IMG_WIDTH
from analysis.leaf_mask import leaf_mask, on_focus
from classification.inference import pred_img
from classification.utils import (
    adaptive_threshold,
    load_model,
    parse_model,
    printArgs,
    set_logging,
    timeSince,
)
from metric import patch_sr, pixel_sr1
from sanity_check.utils import get_saliency_methods, get_saliency_masks
from visualization.viz_helper import (
    get_first_conv_layer,
    get_last_conv_layer,
    normalize_image_attr,
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(2020)

""" Usage
Analyze the full-size leaf disc images and calculate the severity rate and return the saliency map(s)
Given a date, do analysis on all the data collected in that date
"""


parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--model_type', default='VGG', help='model used for training')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model parameters')
parser.add_argument('--loading_epoch', type=int, required=True, help='xth model loaded for inference')
parser.add_argument('--timestamp', required=True, help='model timestamp')
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--model_path', type=str, required=True, help='root path to the model')
parser.add_argument('--step_size', type=int, default=224, help='step size of sliding window')
parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361], help='list of means for each rgb channel')
parser.add_argument('--stds',  type=float, nargs='+', default=[0.144, 0.142, 0.192], help='list of standard deviations for each rgb channel')
parser.add_argument('--target_class', type=int, default=1, help='target class for saliency mapping')
parser.add_argument('--contam_control',  action='store_true', help='use contamination control conditional logic')
parser.add_argument('--pm', type=str, help='PM isolate used for inoculation - collected for metadata in the csv')

# CPU/GPU/MSP parameters
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id', default="0", help='specify cuda id')

# Output parameters
parser.add_argument('--save_infected', action='store_true', help='save infected images')
parser.add_argument('--save_conidiophores', action='store_true', help='save conidiophores images')
parser.add_argument('--save_healthy', action='store_true', help='save healthy images')
parser.add_argument('--sal_threshold', type=float, default=0.5, help='threshold for saliency map')
parser.add_argument('--save_discarded', action='store_true', help='save discarded images')

# Data analysis parameters
parser.add_argument('--up_threshold', type=float, default=0.6, help='upper threshold for severity ratio')
parser.add_argument('--down_threshold', type=float, default=0.2, help='lower threshold for severity ratio')
parser.add_argument('--dataset_path', type=str, required=True, help='root path to the data')
parser.add_argument('--img_folder', type=str, default="2-5-2023_6dpi", help='directory of images')
parser.add_argument('--platform', type=str, default='BlackBird', help='robot platform (Pmbot or BlackBird)')
parser.add_argument('--threshold', nargs='+', help='thresholding value for pixel sr')
parser.add_argument('--log', type=str, default='../../results/logs/random.log', help='log file path')
parser.add_argument('--dpi', type=int, required=True, help='inoculation date')
parser.add_argument('--group', type=str, default='baseline', help='exp group')
parser.add_argument('--trays', nargs='+', required=True, help='tray ids')

# saliency mapping flags
parser.add_argument('--sal_gradcam', action='store_true', help='make saliency map using gradcam')
parser.add_argument('--sal_gradient', action='store_true', help='make saliency map using gradient')
parser.add_argument('--sal_smoothgrad', action='store_true', help='make saliency map using smoothgrad')
parser.add_argument('--sal_deeplift', action='store_true', help='make saliency map using deeplift')
parser.add_argument('--sal_thresh_method', type=str, default='percentile',
                    choices=['percentile', 'fixed'],
                    help='How to compute saliency threshold per image/method')
parser.add_argument('--sal_thresh_p', type=float, default=95.0,
                    help='Percentile used when method=percentile')

opt = parser.parse_args()

# filter out routine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# set device
if opt.cuda and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_id)
    device = torch.device("cuda")
elif opt.mps and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

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
dataset_path = Path(opt.dataset_path) / image_timestamp
mask_path = Path(opt.dataset_path) / f'{image_timestamp}_masking'
model_string = model_type + '_upth' + str(opt.up_threshold) + '_downth' + str(
    opt.down_threshold) + '_' + opt.timestamp
output_folder = Path(opt.dataset_path).parents[0] / 'results' / 'plot_sal_map_output'/ model_string / image_timestamp

# Threshold for severity ratio
down_th = opt.down_threshold  # below this will be classified as healthy
up_th = opt.up_threshold  # above this will be classified as infected or conidiophores
pixel_th = opt.threshold if opt.threshold else []
overlay_thresh_fixed = float(opt.sal_threshold)


rel_th = 0.2  # relative threshold leaf mask
target_class = int(opt.target_class) if opt.target_class != 'None' else None
step_size = opt.step_size

# Model
model_para = parse_model(opt)
model, device = load_model(model_para)
model.eval()
last_conv_layer = get_last_conv_layer(model)
first_conv_layer = get_first_conv_layer(model)

# Normalization
means = opt.means
stds = opt.stds

# Input preprocessing transformation
if opt.model_type == 'Inception3':
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 299
else:
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 224

# Captum
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

# Write severity ratio as CSV files
key = [f'{x}_sr2' for x in saliency_methods.keys()]

if outdim == 3:
    META_COL_NAMES = ['timestamp', 'model_type', 'model_timestamp', 'classes', 'step_size', 'imaging_date', 'tray', 'filename', 'up_th', 'down_th', 'sal_threshold', 'clear_patches', 'hyphal_patches', 'conidiophore_patches',
                  'severity_rate_patch', 'PM', 'time_elapsed'] + key

else:
    META_COL_NAMES = ['timestamp', 'model_type', 'model_timestamp', 'classes', 'step_size', 'imaging_date', 'tray', 'filename', 'up_th', 'down_th', 'sal_threshold', 'clear_patches', 'hyphal_patches',
                      'severity_rate_patch', 'PM', 'time_elapsed'] + key

# List all trays
tray = opt.trays
PM = opt.pm

severity_rate_dfs = defaultdict(lambda: pd.DataFrame(columns=META_COL_NAMES))

threshold = 0.7  # threshold for saliency map
default_cmap = 'Blues'

# Time
total_time = 0
total_time_2 = 0
format_ = 'png'

# Loop trays
for tray_id in opt.trays:
    dataset_tray_path = dataset_path / Path(tray_id)
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path) if x.lower().endswith('.png')]

    # One DF per pixel threshold for this tray
    # If no pixel_th provided, we will fill it adaptively later, but still build dict on the fly.
    per_tray_dfs = {}  # thf(float) -> DataFrame

    for leaf_disk_image_filename in leaf_disk_image_filenames:
        # Per-image timer + timestamp
        start_time = time.time()
        date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

        logger.info('-------------------------------------------')
        logger.info('Processing %s tray=%s file=%s', image_timestamp, tray_id, leaf_disk_image_filename)

        # Load image
        img_filepath = dataset_tray_path / leaf_disk_image_filename
        img = Image.open(img_filepath)
        img_arr = np.asarray(img)
        width, height = img.size

        # Sliding geometry
        subim_x = (width - IMG_WIDTH) // step_size + 1
        subim_y = (height - IMG_HEIGHT) // step_size + 1
        subim_height = (subim_y - 1) * step_size + IMG_HEIGHT
        subim_width = (subim_x - 1) * step_size + IMG_WIDTH
        sub_img = img.crop((0, 0, subim_width, subim_height))
        sub_img_arr = np.asarray(sub_img)

        # Masking
        imask = leaf_mask(img, rel_th=rel_th)
        if imask is None:
            logger.info('Image: %s\tmasking ERROR', imagename_text)
            continue
        imask = (imask.astype('uint8') // 255)  # 0/1

        t1 = time.time()
        logger.info('Finished loading mask: %s', timeSince(start_time))

        # Per-image counters and buffers
        patch_idx = coor_x = coor_y = 0
        infected_patch = conidiophore_patch = clear_patch = discard_patch = lost_focus_patch = 0

        counting_map = np.zeros((height, width), dtype=np.float32)
        prob_attrs1 = np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        if outdim == 3:
            prob_attrs2 = np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        saliency_attrs = {k: np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
                          for k in saliency_methods.keys()}

        # Output folder for this image
        f = imagename_text
        output_leaf_disk_image_folder = output_folder / f'{opt.dpi}_{tray_id}_{f}'
        os.makedirs(output_leaf_disk_image_folder, exist_ok=True)

        # -------------------------------
        # Patch loop
        # -------------------------------
        for _ in range(subim_y):
            for _ in range(subim_x):
                subim_mask = imask[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH]
                if not on_focus(subim_mask):
                    lost_focus_patch += 1
                    prob_attrs1[patch_idx].fill(-np.inf)
                    if outdim == 3:
                        prob_attrs2[patch_idx].fill(-np.inf)
                else:
                    # Crop & preprocess
                    box = (coor_x, coor_y, coor_x + IMG_WIDTH, coor_y + IMG_HEIGHT)
                    subim = img.crop(box).resize((image_width, image_height))
                    subim_arr = np.asarray(subim)
                    input_img = preprocess(subim_arr).unsqueeze(0).to(device).requires_grad_(True)

                    # Inference (keep grads for saliency)
                    pred, prob = pred_img(input_img, model)
                    logits_class = int(pred.item())
                    prob_value = float(prob[0, 1].detach().cpu().item())
                    if outdim == 3:
                        prob_value2 = float(prob[0, 2].detach().cpu().item())

                    # Save probs to buffers
                    prob_attrs1[patch_idx] = prob_value
                    if outdim == 3:
                        prob_attrs2[patch_idx] = prob_value2

                    # Saliency only for predicted positive(s)
                    if (outdim == 2 and logits_class == 1) or (outdim == 3):
                        output_masks = get_saliency_masks(
                            saliency_methods, input_img, logits_class, relu_attributions=True
                        )
                        abs_norm, _, _ = normalize_image_attr(subim_arr, output_masks, hist=False)
                        abs_norm.pop('Original', None)

                        for key, val in abs_norm.items():
                            if image_height != IMG_HEIGHT:
                                # Resize to patch size for reconstruction
                                v = torch.from_numpy(val[None, None, ...])
                                v = F.interpolate(v, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')[0, 0].numpy()
                                saliency_attrs[key][patch_idx] = v
                            else:
                                saliency_attrs[key][patch_idx] = val

                    # Patch classification counts
                    if opt.dpi > 5:
                        if outdim == 3 and prob_value2 >= up_th:
                            conidiophore_patch += 1
                        elif prob_value >= up_th:
                            infected_patch += 1
                        elif prob_value <= down_th:
                            clear_patch += 1
                        else:
                            discard_patch += 1
                    else:
                        if prob_value >= up_th:
                            infected_patch += 1
                        elif outdim == 3 and prob_value2 >= up_th:
                            conidiophore_patch += 1
                        elif prob_value <= down_th:
                            clear_patch += 1
                        else:
                            discard_patch += 1

                            # Save discarded patches
                            if opt.save_discarded:
                                output_leaf_disk_image_folder_discarded = output_leaf_disk_image_folder / 'discarded'
                                if not os.path.exists(output_leaf_disk_image_folder_discarded):
                                    os.makedirs(output_leaf_disk_image_folder_discarded, exist_ok=True)
                                saved_patch_filepath = output_leaf_disk_image_folder_discarded / f'{imagename_text}_image_patch_{patch_idx}_discarded.{format_}'
                                plt.imsave(saved_patch_filepath, subim_arr, cmap=default_cmap, format=format_, dpi=300)

                            # Save healthy patches
                            if opt.save_healthy and logits_class == 0:
                                # print(f"Saving healthy patch for {leaf_disk_image_filename}' class':{logits_class}...")
                                output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / 'clear'
                                if not os.path.exists(output_leaf_disk_image_folder_saliency):
                                    os.makedirs(output_leaf_disk_image_folder_saliency, exist_ok=True)
                                saved_patch_filepath = output_leaf_disk_image_folder_saliency / f'{imagename_text}_image_patch_{patch_idx}_clear.{format_}'
                                plt.imsave(saved_patch_filepath, subim_arr, cmap=default_cmap, format=format_, dpi=300)

                            # Save hyphal patches
                            if opt.save_infected and logits_class == 1:
                                # print(f"Saving infected patch for {leaf_disk_image_filename}' class':{logits_class}...")
                                output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / 'infected'
                                if not os.path.exists(output_leaf_disk_image_folder_saliency):
                                    os.makedirs(output_leaf_disk_image_folder_saliency, exist_ok=True)
                                saved_patch_filepath = output_leaf_disk_image_folder_saliency / f'{imagename_text}_image_patch_{patch_idx}_infected.{format_}'
                                plt.imsave(saved_patch_filepath, subim_arr, cmap=default_cmap, format=format_, dpi=300)

                            # Save conidiophores patches
                            if opt.save_conidiophores and logits_class == 2:  # save infected patches
                                output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / 'conidiophores'
                                if not os.path.exists(output_leaf_disk_image_folder_saliency):
                                    os.makedirs(output_leaf_disk_image_folder_saliency, exist_ok=True)
                                saved_patch_filepath = output_leaf_disk_image_folder_saliency / f'{imagename_text}_image_patch_{patch_idx}_conidiophores.{format_}'
                                plt.imsave(saved_patch_filepath, subim_arr, cmap=default_cmap, format=format_, dpi=300)

                # Update pixel counter each loop to avoid ZeroDivisionError
                counting_map[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0
                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        counting_map[counting_map == 0] = 1.0

        logger.info('Finished crop and inference: %s', timeSince(start_time))

        # -------------------------------
        # Reconstruction to full image
        # -------------------------------
        prob_heatmap1 = np.zeros((height, width), dtype=np.float32)
        if outdim == 3:
            prob_heatmap2 = np.zeros((height, width), dtype=np.float32)
        saliency_heatmaps = {k: np.zeros((height, width), dtype=np.float32) for k in saliency_methods.keys()}

        patch_idx = coor_x = coor_y = 0
        for _ in range(subim_y):
            for _ in range(subim_x):
                prob_heatmap1[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += prob_attrs1[patch_idx]
                if outdim == 3:
                    prob_heatmap2[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += prob_attrs2[patch_idx]
                for k in saliency_methods.keys():
                    saliency_heatmaps[k][coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += saliency_attrs[k][
                        patch_idx]
                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        # Normalize by coverage
        prob_heatmap1 /= counting_map
        if outdim == 3:
            prob_heatmap2 /= counting_map
        for k in saliency_heatmaps:
            saliency_heatmaps[k] /= counting_map

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

        # -------------------------------
        # Severity metrics
        # -------------------------------
        patch_info = {
            'infected_patch': infected_patch,
            'conidiophore_patch': conidiophore_patch,
            'clear_patch': clear_patch,
            'discard_patch': discard_patch,
            'lost_focus_patch': lost_focus_patch,
        }
        heatmap_info = saliency_heatmaps.copy()
        heatmap_info['prob_heatmap1'] = prob_heatmap1
        if outdim == 3:
            heatmap_info['prob_heatmap2'] = prob_heatmap2

        threshold_info = {
            'patch_down_th': down_th,
            'patch_up_th': up_th,
            'pixel_th': [float(x) for x in pixel_th] if pixel_th else [overlay_thresh_fixed],
        }

        if outdim == 3:
            severity_rate_patch, _ = patch_sr.metric_two_class(patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, _ = pixel_sr1.metric(patch_info.copy(), heatmap_info.copy(), threshold_info.copy(),
                                                       outdim)
        else:
            severity_rate_patch, _ = patch_sr.metric(patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, _ = pixel_sr1.metric(patch_info.copy(), heatmap_info.copy(), threshold_info.copy(),
                                                       outdim)

        if opt.contam_control and opt.dpi > 6 and outdim == 3 and conidiophore_patch < 2 and infected_patch > 10:
            infected_patch = "NA"
            conidiophore_patch = "NA"

        # -------------------------------
        # Visualizations (overlay uses per-method adaptive t)
        # -------------------------------
        alpha = 0.5

        # raw
        out_fp = output_leaf_disk_image_folder / f'{opt.dpi}_{f}_raw.{format_}'
        plt.imshow(img_arr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_fp, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # masked background
        sub_img_arr_copy = img_arr.copy()
        sub_img_arr_copy[imask == 0] = 0
        sub_img_arr_copy = (sub_img_arr_copy.astype('uint8') / 255)

        # Masked leaf disk
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                                          f'{opt.dpi}_{f}_masked.{format_}'

        sub_img_arr_copy = img_arr.copy()
        sub_img_arr_copy[imask == 0] = 0
        sub_img_arr_copy = sub_img_arr_copy.astype('uint8') / 255
        plt.imshow(sub_img_arr_copy)
        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(output_leaf_disk_image_filepath, format=format_,
        #            dpi=300, bbox_inches='tight', pad_inches=0)

        if outdim == 2:
            # For class 1
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / f'{opt.dpi}_{f}_patch_based_class1.{format_}'
            value = prob_heatmap1.copy()
            value[value < up_th] = 0
            value[value >= up_th] = 1
            value = value.astype('uint8')
            # Count patches with value = 0 and value = 1
            count_0 = np.sum(value == 0) / (224 * 224)
            count_1 = np.sum(value == 1) / (224 * 224)
            alphas = np.full(imask.shape, alpha)
            alphas[value == 0] = 0
            plt.imshow(value, alpha=alphas, cmap=default_cmap)
            # Display counts on the figure
            plt.text(100, 300, f'Healthy Patches: {clear_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 500, f'Hyphal Patches: {infected_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 700, f'Discarded Patches: {discard_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 900, f'Infection Severity Rate: {severity_rate_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        if outdim == 3:
            # For both classes
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / f'{opt.dpi}_{f}_patch_based_both_classes.{format_}'
            value = prob_heatmap1.copy()
            value2 = prob_heatmap2.copy()
            value[value < up_th] = 0
            value2[value2 < up_th] = 0
            combined_value = value + value2
            combined_value[combined_value > 0] = 1
            combined_value = combined_value.astype('uint8')

            alphas = np.full(imask.shape, alpha)
            alphas[combined_value == 0] = 0
            plt.imshow(sub_img_arr_copy)  # display the original image

            plt.imshow(value, alpha=alphas, cmap=default_cmap)
            plt.imshow(value2, alpha=alphas, cmap=default_cmap)

            value = value.astype('uint8')
            value2 = value2.astype('uint8')

            # Count patches for each class
            count_class1 = np.sum(value == 1)
            count_class2 = np.sum(value2 == 1)
            count_combined = np.sum(combined_value == 1) / (224 * 224)

            # Display information on the figure
            plt.text(100, 300, f'Healthy Patches: {clear_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 500, f'Hyphal Patches: {infected_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 700, f'Conidiophore Patches: {conidiophore_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 900, f'Total Infected Patches: {count_combined}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 1100, f'Discarded Patches: {discard_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 1300, f'Infection Severity Rate: {severity_rate_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # per-method overlays
        for key, heat in saliency_heatmaps.items():
            t = float(adaptive_th.get(key, overlay_thresh_fixed))
            bin_map = (heat >= t).astype('uint8')
            out_fp = output_leaf_disk_image_folder / f'{opt.dpi}_{key}_th{t:.4f}_{f}_blended.{format_}'
            alphas = np.full(imask.shape, alpha, dtype=float)
            alphas[bin_map == 0] = 0.0
            plt.imshow(sub_img_arr_copy)
            plt.imshow(bin_map, alpha=alphas, cmap=default_cmap)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_fp, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            out_fp = output_leaf_disk_image_folder / f'{key}_sal_bin_th{t:.4f}_{f}.{format_}'
            plt.imshow(bin_map, cmap=default_cmap)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_fp, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # -------------------------------
        # Record rows for each pixel threshold
        # -------------------------------
        for thf in threshold_info['pixel_th']:
            if thf not in per_tray_dfs:
                per_tray_dfs[thf] = pd.DataFrame(columns=META_COL_NAMES)

            if outdim == 3:
                row = [
                          date_time_str, model_type, model_timestamp, outdim, step_size, image_timestamp, tray_id,
                          imagename_text, up_th, down_th, thf, clear_patch, infected_patch, conidiophore_patch,
                          severity_rate_patch, PM, timeSince(start_time)
                      ] + list(severity_rates_pixel[float(thf)].values())
                per_tray_dfs[thf] = pd.concat([per_tray_dfs[thf], pd.DataFrame([row], columns=META_COL_NAMES)],
                                              ignore_index=True)
            else:
                row = [
                          date_time_str, model_type, model_timestamp, outdim, step_size, image_timestamp, tray_id,
                          imagename_text, up_th, down_th, thf, clear_patch, infected_patch,
                          severity_rate_patch, PM, timeSince(start_time)
                      ] + list(severity_rates_pixel[float(thf)].values())
                per_tray_dfs[thf] = pd.concat([per_tray_dfs[thf], pd.DataFrame([row], columns=META_COL_NAMES)],
                                              ignore_index=True)

        logger.info('Analysis finished: %s', timeSince(start_time))
        logger.info('-------------------------------------------')

        # Free per-image memory
        del img, img_arr, sub_img, sub_img_arr, prob_heatmap1, saliency_heatmaps
        gc.collect()

    # -------------------------------
    # Save CSVs for this tray
    # -------------------------------
    output_csv_folder_th = output_folder / 'th'
    os.makedirs(output_csv_folder_th, exist_ok=True)
    for thf, df in per_tray_dfs.items():
        out_csv = output_csv_folder_th / f'severity_rate_tray{tray_id}_th{thf:.1f}.csv'
        df.to_csv(out_csv, index=False)
        logger.info('Saved %s', out_csv)
