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
    timeSince
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
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--model_path', type=str, required=True, help='root path to the model')
parser.add_argument('--step_size', type=int, default=224, help='step size of sliding window')
parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361],
                    help='List of means for each channel')
parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192],
                    help='List of standard deviations for each channel')
parser.add_argument('--target_class', type=int, default=1, help='target class for saliency mapping')
parser.add_argument('--contam_control', action='store_true', help='use contamination control conditional logic')

# CPU/GPU/MSP parameters
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id', default="0", help='specify cuda id')

# Output parameters
parser.add_argument('--sal_threshold', type=float, default=0.5, help='threshold for saliency map')

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
parser.add_argument('--trays', nargs='+', help='trays')
parser.add_argument('--pm', type=str, help='pm isolate for metadata')

# filter out routine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# saliency mapping flags
parser.add_argument('--sal_gradcam', action='store_true')
parser.add_argument('--sal_gradient', action='store_true')
parser.add_argument('--sal_smoothgrad', action='store_true')
parser.add_argument('--sal_deeplift', action='store_true')

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
                  'infected_threshold', 'healthy_threshold', 'sal_threshold', 'leaf_mask_th', 'clear_patches',
                  'infected_patches'] + \
                 (['conidiophore_patches', 'sporulating_pct'] if outdim == 3 else []) + \
                 ['discarded_patches', 'severity_rate_patch'] + key

# List all trays
tray = opt.trays
PM = opt.pm

#threshold = 0.7  # threshold for saliency map

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
        if outdim == 3:
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
                    if outdim == 3:
                        prob_attrs2[patch_idx] = -np.inf

                else:
                    # Cropping
                    box = (coor_x, coor_y, coor_x + IMG_WIDTH, coor_y + IMG_HEIGHT)
                    subim = img.crop(box).resize((image_width, image_height))
                    subim_arr = np.asarray(subim)

                    # Preprocess
                    input_img = preprocess(subim_arr).unsqueeze(0).to(device)

                    # Forward pass
                    with torch.no_grad():
                        pred, prob = pred_img(input_img, model, track_grad=False, use_autocast=True)

                    # Probability of infected (class 1)
                    prob_value = prob[0, 1].item()
                    prob_attrs1[patch_idx] = prob_value

                    # Predicted class index (works whether pred is 1D or 2D)
                    logits_class = int(prob.argmax(dim=-1).item())

                    # If we are in the 3-class setup, also extract conidiophore probability
                    if outdim == 3:
                        prob_value2 = prob[0, 2].item()
                        prob_attrs2[patch_idx] = prob_value2

                    # Store original predicted class (used later for saliency logic)
                    original_logits_class = logits_class

                    # --- Saliency (only if any saliency method is enabled) ---
                    if saliency_methods:
                        saliency_class = int(logits_class)  # which class we explain

                        # Only compute saliency for relevant disease classes:
                        #   3-class model: only infected (1) or conidiophore (2)
                        #   2-class model: only infected (1)
                        if (outdim == 3 and saliency_class in (1, 2)) or (outdim == 2 and saliency_class == 1):

                            # Make input require gradients only while computing saliency
                            input_img.requires_grad_(True)

                            output_masks = get_saliency_masks(
                                saliency_methods,
                                input_img,
                                saliency_class,
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

                            input_img.requires_grad_(False)  # turn gradients off again

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

        threshold_info = {'patch_down_th': down_th, 'patch_up_th': up_th, 'pixel_th': [float(th)]}

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
        discard_patches = discard_patch
        discarded_patches = patch_info['discard_patch']

        if opt.contam_control and opt.dpi > 6 and outdim == 3 and conidiophore_patch < 2 and infected_patch > 10:
            infected_patch = "NA"
            conidiophore_patch = "NA"  # this is to catch contamination

        # % sporulating (only if outdim=3). Respect "NA" from contamination control and avoid divide-by-zero.
        sporulating_pct = None
        if outdim == 3:
            if isinstance(infected_patch, str) or isinstance(conidiophore_patch, str):
                sporulating_pct = "NA"
            else:
                total_patches = clear_patch + infected_patch + conidiophore_patch
                sporulating_pct = (conidiophore_patch / total_patches * 100.0) if total_patches > 0 else np.nan

        for i, th in enumerate(pixel_th):
            # Extract conserved_identifier from imagename_text
            conserved_identifier = imagename_text[4:].split('_')[0]

            record_data = [
                date_time_str, timeSince(start_time), model_type, model_timestamp, outdim,
                image_timestamp, tray_id, imagename_text, conserved_identifier, '', '', '', PM,
                up_th, down_th, float(th), rel_th, clear_patch, infected_patch
            ]

            if outdim == 3:
                record_data.append(conidiophore_patch)
                record_data.append(sporulating_pct)

            record_data += [discarded_patches, severity_rate_patch] + list(severity_rates_pixel[float(th)].values())
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
        out_path = output_csv_folder_th / f'severity_rate_tray_{tray_id}.csv'
        severity_rate_df_list[i].to_csv(out_path, index=False)
        logger.info('Saved %s', out_path)

    # Explicitly delete objects to free memory
    del img, img_arr, sub_img, sub_img_arr, prob_heatmap1, saliency_heatmaps  # prob_heatmap2
    # Call garbage collector
    gc.collect()
