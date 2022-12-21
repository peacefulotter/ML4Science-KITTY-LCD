import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

def RRE(Rt, Re):
    '''
    Computes the Relative Rotational Error (RRE)
    given two rotation matrices (ground truth and estimated)

    Params:
        - (3, 3) Rt: ground-truth rotation matrix
        - (3, 3) Re: estimated rotation matrix

    Returns: The relative rotation error
    '''
    angles = np.linalg.inv(Rt) @ Re 
    angles = R.from_matrix(angles)
    angles = angles.as_euler('zyx', degrees=True)
    return np.sum(np.abs(angles))

def RTE(Tt, Te):
    '''
    Computes the Relative Translation Error (RTE)
    given two translation vectors (ground truth and estimated)

    Params:
        - (3, 1) Tt: ground-truth translation vector
        - (3, 1) Te: estimated translation vector

    Returns: The relative translation error
    '''
    return np.linalg.norm(Tt - Te)

def patch_coordinates(origin, patch_size):
    '''
    Get the coordinates corresponding to a patch with origin coordinates 'origin'
    and size 'patch_size'

    Params:
        - (2, ) origin: tuple of origin coordinates (y, x)
        - (2, ) patch_size: tuple defining the patch size (h, w) 

    Returns: 
        - (h * w, 2) patch_coords: patch coordinates as floats

    Example:
    >>> origin = (1, 1)
    >>> patch_size = (2, 3)
    >>> [[1., 1.],
        [2., 1.],
        [1., 2.],
        [2., 2.],
        [1., 3.],
        [2., 3.]]
    '''
    o_h, o_w = origin
    patch_h, patch_w = patch_size
    y = np.arange(o_h, o_h + patch_h)
    x = np.arange(o_w, o_w + patch_w)
    g = np.meshgrid(y, x) # TODO
    patch_coords = np.vstack(map(np.ravel, g)).T
    return patch_coords.astype(float)

@torch.no_grad()
def get_estimated_pose(pc, K, origin, patch_size, dist_thres=5): # 5 pixels
    '''
    Get estimated pose using PnPRansac and Rodrigues.

    Params:
        - (N, 6) pc: RGB colored pointcloud (works also with a pc of shape (N, 3))
        - (3, 3) K: Intrinsic camera parameters
        - (2, ) origin: tuple of origin coordinates (y, x)
        - (2, ) patch_size: tuple defining the patch size (h, w) 
        - dist_thres (optional): reprojection error for PnP in pixels, default to 5

    Returns: 
        - (3, 3) R: Rotation matrix
        - (3, 1) t: Translation vector
    '''
    pc = pc.cpu().detach().numpy().astype(float)
    pc = pc[:, :3] # keep only xyz
    img = patch_coordinates(origin, patch_size)
    print(pc.shape, pc.dtype)
    print(img.shape, img.dtype)

    # Works only because img_h * img_w = 4 * num_pc = 4 * 1024
    pc = np.repeat(pc, 4)
    print(pc.shape)

    _, R, t, _ = cv2.solvePnPRansac(
        objectPoints=pc, 
        imagePoints=img, 
        cameraMatrix=K, 
        useExtrinsicGuess=False,
        iterationsCount=500, 
        reprojectionError=dist_thres, 
        flags=cv2.SOLVEPNP_EPNP, 
        distCoeffs=None
    )
    R, _ = cv2.Rodrigues(R) # Converts rotation vector to matrix
    return R, t
