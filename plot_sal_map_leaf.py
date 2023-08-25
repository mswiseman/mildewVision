import os
import time
import argparse
import pandas as pd
import numpy as np
import warnings
import gc

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

from analyzer_config import (CHANNELS, IMG_HEIGHT, IMG_WIDTH, IMG_EXT, INPUT_SIZE)

from metric import pixel_sr1, patch_sr

from classification.inference import pred_img
from classification.utils import timeSince, printArgs, load_model, parse_model, set_logging

from analysis.leaf_mask import leaf_mask, on_focus

from visualization.viz_util import _normalize_image_attr
from visualization.viz_helper import (get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr,
                                      plot_figs, save_figs)

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
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--model_path', type=str, required=True, help='root path to the model')
parser.add_argument('--step_size', type=int, default=224, help='step size of sliding window')
parser.add_argument('--means', type=float, nargs='+', default="0.504 0.604 0.361",
                    help='List of means for each channel')
parser.add_argument('--stds', type=float, nargs='+', default="0.144 0.142 0.192",
                    help='List of standard deviations for each channel')
parser.add_argument('--target_class', type=int, default=1, help='target class for saliency mapping')

# CPU/GPU/MSP parameters
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id', default="0", help='specify cuda id')

# Output parameters
parser.add_argument('--save_infected', action='store_true', help='save infected images')
parser.add_argument('--save_conidiophores', action='store_true', help='save conidiophores images')
parser.add_argument('--save_healthy', action='store_true', help='save healthy images')

# Data analysis parameters
parser.add_argument('--up_threshold', type=float, default=0.6, help='upper threshold for severity ratio')
parser.add_argument('--down_threshold', type=float, default=0.2, help='lower threshold for severity ratio')
parser.add_argument('--dataset_path', type=str, required=True, help='root path to the data')
parser.add_argument('--img_folder', type=str, default="2-5-2023_6dpi", help='directory of images')
parser.add_argument('--platform', type=str, default='PMbot', help='robot platform (Pmbot or BlackBird)')
parser.add_argument('--threshold', nargs='+', help='thresholding value for pixel sr')
parser.add_argument('--log', type=str, default='../../results/logs/random.log', help='log file path')
parser.add_argument('--dpi', type=int, required=True, help='inoculation date')
parser.add_argument('--group', type=str, default='baseline', help='exp group')
parser.add_argument('--trays', nargs='+', help='trays')
opt = parser.parse_args()

# filter out routine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# set device
if opt.mps:
    device_type = "mps"
elif opt.cuda:
    device_type = "cuda"
else:
    device_type = "cpu"

gpu = torch.device(device_type)

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
output_folder = Path(opt.dataset_path).parents[0] / 'Results' / model_string / image_timestamp

# Threshold for severity ratio
down_th = opt.down_threshold  # below this will be classified as healthy
up_th = opt.up_threshold  # above this will be classified as infected or conidiophores
pixel_th = opt.threshold if opt.threshold else []

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
                                        gradcam=True,
                                        gradient=True,
                                        smooth_grad=True,
                                        deeplift=True)

# Prepare columns and key for later csv file
key = [f'{x}_sr2' for x in saliency_methods.keys()]
META_COL_NAMES = ['model_type', 'model_timestamp', 'classes', 'step_size', 'timestamp', 'tray', 'filename', 'up_th', 'down_th', 'sal_threshold', 'clear_patches', 'hyphal_patches', 'conidiophore_patches',
                  'severity_rate_patch', 'time_elapsed'] + key

# List all trays
tray = opt.trays

threshold = 0.2 # threshold for saliency map
default_cmap = 'Blues'

# Time
total_time = 0
total_time_2 = 0
format_ = 'png'

# Loop through trays
for trays in tray:
    dataset_tray_path = dataset_path / Path(tray[0])
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path) if x.endswith('.png')]

    severity_rate_df_list = []
    for th in pixel_th:
        severity_rate_df_list.append(pd.DataFrame(columns=META_COL_NAMES))

    # Loop through leaf disk images within tray files
    for leaf_disk_image_filename in leaf_disk_image_filenames:
        img_filepath = dataset_tray_path / leaf_disk_image_filename

        # Timer
        start_time = time.time()

        logger.info('-------------------------------------------')
        logger.info('Processing {} {} {}'.format(
            image_timestamp, tray, leaf_disk_image_filename))

        # Get info of resized image subim_x: number of patches one row
        img = Image.open(img_filepath)
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
        if outdim == 3:
            prob_attrs2 = np.zeros(
                shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=float)

        saliency_attrs = {}
        for saliency_method_key in saliency_methods.keys():
            saliency_attrs[saliency_method_key] = np.zeros(
                shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=float)

        f = os.path.splitext(leaf_disk_image_filename)[0]
        output_leaf_disk_image_folder = output_folder / f'{opt.dpi}_{tray}_{f}'
        if not os.path.exists(output_leaf_disk_image_folder):
            os.makedirs(output_leaf_disk_image_folder, exist_ok=True)

        # Crop
        for _ in range(subim_y):
            for _ in range(subim_x):
                subim_mask = imask[coor_y: coor_y + IMG_HEIGHT, coor_x: coor_x + IMG_WIDTH]
                if not on_focus(subim_mask):
                    # Set lost focused patches' pixel values as -inf
                    lost_focus_patch += 1
                    prob_attrs1[patch_idx] = -np.inf
                    if outdim == 3:
                        prob_attrs2[patch_idx] = -np.inf

                else:
                    # Cropping
                    box = (coor_x, coor_y, coor_x + IMG_WIDTH, coor_y + IMG_HEIGHT)
                    subim = img.crop(box).resize((image_width, image_height))
                    subim_arr = np.asarray(subim)

                    # Preprocess
                    input_img = preprocess(subim_arr).unsqueeze(0).to(device)
                    input_img.requires_grad = True

                    # Forward pass
                    pred, prob = pred_img(input_img, model)
                    prob_value = prob[0][1].cpu().detach().item()  # Probability of class 1
                    logits_class = pred.cpu().detach().item()  # Update logits_class based on the predicted class

                    # Save to the entire array
                    prob_attrs1[patch_idx] = prob[0][1].cpu().detach().item()  # Probability of class 1
                    if outdim == 3:
                        prob_value2 = prob[0][2].cpu().detach().item()  # Probability of class 2
                        prob_attrs2[patch_idx] = prob[0][2].cpu().detach().item()  # Probability of class 2

                    # Store the original logits class
                    original_logits_class = logits_class

                    if outdim == 3:
                        for saliency_class in [1, 2]:
                            output_masks = get_saliency_masks(
                                saliency_methods, input_img, logits_class, relu_attributions=True)

                            # Save to the entire array
                            prob_attrs1[patch_idx] = prob[0][1].cpu().detach().item()  # Probability of class 1
                            prob_attrs2[patch_idx] = prob[0][2].cpu().detach().item()  # Probability of class 2

                            # Normalization
                            abs_norm, no_abs_norm, _0_1_norm = normalize_image_attr(
                                subim_arr, output_masks, hist=False)
                            abs_norm.pop('Original')

                            for key, val in abs_norm.items():
                                if image_height != IMG_HEIGHT:
                                    # Adapt to the F.interpolate() API
                                    val = torch.from_numpy(
                                        val[np.newaxis, np.newaxis, ...])
                                    val = F.interpolate(
                                        val, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')[0][0]
                                    saliency_attrs[key][patch_idx] = val

                                else:
                                    saliency_attrs[key][patch_idx] = val

                    if outdim == 2:
                        for saliency_class in [1]:
                            # for logits_class in [1]:
                            output_masks = get_saliency_masks(
                                saliency_methods, input_img, logits_class, relu_attributions=True)

                            # Save to the entire array
                            prob_attrs1[patch_idx] = prob[0][1].cpu().detach().item()  # Probability of class 1

                            # Normalization
                            abs_norm, no_abs_norm, _0_1_norm = normalize_image_attr(
                                subim_arr, output_masks, hist=False)
                            abs_norm.pop('Original')

                            for key, val in abs_norm.items():
                                if image_height != IMG_HEIGHT:
                                    # Adapt to the F.interpolate() API
                                    val = torch.from_numpy(
                                        val[np.newaxis, np.newaxis, ...])
                                    val = F.interpolate(
                                        val, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')[0][0]
                                    saliency_attrs[key][patch_idx] = val
                                else:
                                    saliency_attrs[key][patch_idx] = val

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

                    # This logic separates preferential treatment of classes by days post inoculation
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
        if outdim == 3:
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
                if outdim == 3:
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
        if outdim == 3:
            prob_heatmap2 = prob_heatmap2 / counting_map

        for key, val in saliency_heatmaps.items():
            saliency_heatmaps[key] = val / counting_map
            # print("key: ", key, "val: ", val)

        # Severity rate calculation
        patch_info = {'infected_patch': infected_patch, 'conidiophore_patch': conidiophore_patch,
                      'clear_patch': clear_patch, 'discard_patch': discard_patch, 'lost_focus_patch': lost_focus_patch}
        heatmap_info = saliency_heatmaps.copy()
        heatmap_info['prob_heatmap1'] = prob_heatmap1

        threshold_info = {'patch_down_th': down_th,
                          'patch_up_th': up_th, 'pixel_th': [threshold]}

        if outdim == 3:
            heatmap_info['prob_heatmap2'] = prob_heatmap2

        # Calculate severity rate
        # print(f"Value of outdim: {outdim}")
        if outdim == 3:
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

        # Visualization
        alpha = 0.5

        # Raw leaf disk
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                                          f'{opt.dpi}_{f}_raw.{format_}'
        plt.imshow(img_arr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_leaf_disk_image_filepath, format=format_,
                    dpi=300, bbox_inches='tight', pad_inches=0)

        # Masked leaf disk
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                                          f'{opt.dpi}_{f}_masked.{format_}'

        sub_img_arr_copy = img_arr.copy()
        sub_img_arr_copy[imask == 0] = 0
        sub_img_arr_copy = sub_img_arr_copy.astype('uint8') / 255
        plt.imshow(sub_img_arr_copy)
        plt.axis('off')
        plt.tight_layout()
        #plt.savefig(output_leaf_disk_image_filepath, format=format_,
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
            plt.text(100, 370, f'Healthy Patches: {clear_patch}', color='white', fontsize=6)
            plt.text(100, 710, f'Hyphal Patches: {infected_patch}', color='white', fontsize=6)
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
            discard_patches = discard_patch

            # Display information on the figure
            plt.text(100, 300, f'Healthy Patches: {clear_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 500, f'Hyphal Patches: {infected_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 700, f'Conidiophore Patches: {conidiophore_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 900, f'Total Infected Patches: {count_combined}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 1100, f'Discarded Patches: {discard_patches}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))
            plt.text(100, 1300, f'Infection Severity Rate: {severity_rate_patch}', color='white', fontsize=6,
                     bbox=dict(facecolor='black', alpha=0.5))

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Heatmap
        for key, value in saliency_heatmaps.items():
            value[value < threshold] = 0
            value[value >= threshold] = 1
            value = value.astype('uint8')

            # Overlap with original image
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                                              f'{opt.dpi}_{key}_{f}_blended.{format_}'
            alphas = np.full(imask.shape, alpha)
            alphas[value == 0] = 0
            plt.imshow(sub_img_arr_copy)
            plt.imshow(value, alpha=alphas, cmap=default_cmap)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_,
                        dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.imshow(value, cmap=default_cmap)
            plt.axis('off')
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                                              f'{key}_sal_th_{threshold}_{f}.{format_}'
            # Only heatmap
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_,
                        dpi=300, bbox_inches='tight', pad_inches=0)

        for i, th in enumerate(pixel_th):
            # Create the DataFrame without specifying columns
            record_data = [model_type, model_timestamp, outdim, step_size, image_timestamp, tray, imagename_text, up_th, down_th, threshold, clear_patch, infected_patch, conidiophore_patch,
                           severity_rate_patch, timeSince(start_time)] + list(
                severity_rates_pixel[float(th)].values())
            # print("list(severity_rates_pixel[float(th)].values()): ", list(severity_rates_pixel[float(th)].values()))
            # print("record_data: ", record_data)
            record_df = pd.DataFrame([record_data], columns=META_COL_NAMES)

            severity_rate_df_list[i] = pd.concat([severity_rate_df_list[i], record_df], ignore_index=True)

            output_csv_folder_th = output_folder / f'th'
            if not os.path.exists(output_csv_folder_th):
                os.makedirs(output_csv_folder_th, exist_ok=True)

            output_csv_filepath = output_csv_folder_th / 'severity_rate.csv'
            # print("output_csv_filepath: ", output_csv_filepath)

            severity_rate_df_list[i].to_csv(output_csv_filepath, index=False)
            logger.info('Saved {}'.format(output_csv_filepath))

        total_time = total_time + time.time() - start_time
        total_time_2 = total_time_2 + time.time() - t1
        logger.info('Analysis finished: {}'.format(timeSince(start_time)))
        logger.info('-------------------------------------------')

        # Explicitly delete objects to free memory
        del img, img_arr, sub_img, sub_img_arr, prob_heatmap1, prob_heatmap2, saliency_heatmaps
        # Call garbage collector
        gc.collect()
