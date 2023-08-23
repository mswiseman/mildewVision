import sys
import numpy as np

sys.path.append('..')

from utils import hard_thresholding, otsu_thresholding
from analyzer_config import IMG_HEIGHT, IMG_WIDTH

def metric(patch_info, heatmap_info, threshold_info, outdim):

    """
    Calculate pixel-level severity rate using a gradcam heatmap.

    Args:
    - patch_info (dict): Contains patch-related information.
    - heatmap_info (dict): Contains heatmap-related information.
    - threshold_info (dict): Contains threshold-related information.
    - outdim (int): Output dimension of the model.

    Returns:
    - dict: Severity rates for each threshold and key.
    - tuple: Infected pixels and total pixels.
    """
    # Remove 'prob_heatmap[x]' if exists
    heatmap_info.pop('prob_heatmap', None)
    heatmap_info.pop('prob_heatmap1', None)
    heatmap_info.pop('prob_heatmap2', None)

    lost_focus_patch = patch_info['lost_focus_patch']
    lost_focus_pixel = patch_info['lost_focus_patch'] * IMG_HEIGHT * IMG_WIDTH
    pixel_th = threshold_info['pixel_th']
    # print("lost_focus_patch: ", lost_focus_patch)


    severity_rates = {}
    infected_pixels = {}
    total_pixels = {}

    for th in pixel_th:
        for key, val in heatmap_info.items():
            saliency_mask = otsu_thresholding(val, vmin=0, vmax=1) if th == 'otsu' else hard_thresholding(val,
                                                                                                          float(th),
                                                                                                          vmin=0,
                                                                                                          vmax=1)

            total_pixel = saliency_mask.size
            clear_pixel = np.sum(saliency_mask == 0)
            infected_pixel = np.sum(saliency_mask == 1) + (np.sum(saliency_mask == 2) if outdim == 3 else 0)

            assert clear_pixel + infected_pixel == total_pixel

            severity_rates.setdefault(th, {})[key] = round(infected_pixel / (total_pixel - lost_focus_pixel) * 100, 2)
            infected_pixels.setdefault(th, {})[key] = infected_pixel
            total_pixels.setdefault(th, {})[key] = total_pixel - lost_focus_pixel

            # print("key: ", key)
            # print("severity_rates: ", severity_rates)

        return severity_rates, (infected_pixel, total_pixel)
