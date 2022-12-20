
import numpy as np
from sklearn.feature_extraction.image import _extract_patches, _compute_n_patches

'''
Custom sklearn.feature_extraction.image.extract_patches_2d
to return the patches origin coordinates as well as the patches
'''

def extract_patches_2d(image, patch_size, max_patches):
    i_h, i_w, n_colors = image.shape
    p_h, p_w = patch_size

    extracted_patches = _extract_patches(
        image, patch_shape=(p_h, p_w, n_colors), extraction_step=1
    )

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    i_s = np.random.randint(i_h - p_h + 1, size=n_patches)
    j_s = np.random.randint(i_w - p_w + 1, size=n_patches)
    patches = extracted_patches[i_s, j_s, 0]

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    origins = np.vstack((i_s, j_s)).T

    return patches, origins