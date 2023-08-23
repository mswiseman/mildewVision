import sys
import numpy as np

sys.path.append('..')

from analyzer_config import IMG_HEIGHT, IMG_WIDTH


def metric(patch_info, heatmap_info, threshold_info):
    """
        Calculate patch level severity rate using probability heatmap
    """
    prob_heatmap1 = heatmap_info['prob_heatmap1']
    patch_down_th = threshold_info['patch_down_th']
    patch_up_th = threshold_info['patch_up_th']

    infected_patch = patch_info['infected_patch']
    clear_patch = patch_info['clear_patch']

    infected_pixel1 = len(prob_heatmap1[prob_heatmap1 >= patch_up_th])
    clear_pixel1 = len(prob_heatmap1[prob_heatmap1 <= patch_down_th]) - len(prob_heatmap1[prob_heatmap1 == -np.inf])

    # Uncomment for troubleshooting
    # print("patch_up_th: ", patch_up_th)
    # print("patch_down_th: ", patch_down_th)
    # print("infected_pixel: ", infected_pixel)
    # print("clear_pixel: ", clear_pixel)
    # print("infected_patch: ", infected_patch)
    # print("conidiophores_patch", conidiophores_patch)
    # print("clear_patch: ", clear_patch)
    # print("patch_sr: ", round((infected_patch / (infected_patch + clear_patch)) * 100, 2))

    return round((infected_patch / (infected_patch + clear_patch)) * 100, 2), (infected_pixel1, clear_pixel1)


def metric_two_class(patch_info, heatmap_info, threshold_info):
    prob_heatmap1 = heatmap_info['prob_heatmap1']  # infected class probability heatmap
    prob_heatmap2 = heatmap_info['prob_heatmap2']  # conidiophore class probability heatmap

    patch_down_th = threshold_info['patch_down_th']
    patch_up_th = threshold_info['patch_up_th']

    infected_patch = patch_info['infected_patch']
    conidiophores_patch = patch_info['conidiophores_patch']
    clear_patch = patch_info['clear_patch']

    # Process prob_heatmap1
    infected_pixel1 = len(prob_heatmap1[prob_heatmap1 >= patch_up_th])
    clear_pixel1 = len(prob_heatmap1[prob_heatmap1 <= patch_down_th]) - len(prob_heatmap1[prob_heatmap1 == -np.inf])

    # Process prob_heatmap2
    infected_pixel2 = len(prob_heatmap2[prob_heatmap2 >= patch_up_th])
    clear_pixel2 = len(prob_heatmap2[prob_heatmap2 <= patch_down_th]) - len(prob_heatmap2[prob_heatmap2 == -np.inf])

    # Calculate infected_pixel and clear_pixel
    infected_pixel = (infected_pixel1 + infected_pixel2) / (
            clear_pixel1 + clear_pixel2 + infected_pixel1 + infected_pixel2)
    clear_pixel = (clear_pixel1 + clear_pixel2) / (clear_pixel1 + clear_pixel2 + infected_pixel1 + infected_pixel2)

    # Uncomment for troubleshooting
    # print("patch_up_th: ", patch_up_th)
    # print("patch_down_th: ", patch_down_th)
    # print("infected_pixel: ", infected_pixel)
    # print("clear_pixel: ", clear_pixel)
    # print("infected_patch: ", infected_patch)
    # print("conidiophores_patch", conidiophores_patch)
    # print("clear_patch: ", clear_patch)
    # print("patch_sr: ", round(((infected_patch + conidiophores_patch) / (infected_patch + clear_patch + conidiophores_patch)) * 100, 2))

    # returns the percentage of infected pixels and the number of infected and clear pixels
    return round(((infected_patch + conidiophores_patch) / (infected_patch + conidiophores_patch + clear_patch)) * 100, 2), (infected_pixel, clear_pixel)
