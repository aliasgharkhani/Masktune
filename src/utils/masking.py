import numpy as np

def mask_heatmap_using_threshold(heat_maps):
    mask_mean_value = np.nanmean(
        np.where(heat_maps > 0, heat_maps, np.nan), axis=(1, 2))[:, None, None]
    mask_std_value = np.nanstd(
        np.where(heat_maps > 0, heat_maps, np.nan), axis=(1, 2))[:, None, None]
    mask_threshold_value = mask_mean_value + 2 * mask_std_value
    return np.expand_dims(
        np.where(heat_maps > mask_threshold_value, 0, 1), axis=-1)
