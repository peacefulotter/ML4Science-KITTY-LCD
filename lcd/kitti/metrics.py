import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

def RRE(Rt, Re):
    '''
    params
        Rt: ground-truth rotation matrix, shape=(3, 3)
        Re: estimated rotation matrix, shape=(3, 3)

    returns:
        RRE (Relative Rotational Error) \n
        rre = sum(abs(F(Rt^(-1) @ Re))) \n
        where F(.) transforms a matrix to three Euler angle
    '''
    angles = np.linalg.inv(Rt) @ Re 
    angles = R.from_matrix(angles)
    angles = angles.as_euler('zyx', degrees=True)
    return np.sum(np.abs(angles))

def RTE(Tt, Te):
    '''
    params
        Tt: ground-truth translation vector, shape=(3, 1)
        Te: estimated translation vector, shape=(3, 1)

    returns:
        RTE (Relative Translational Error) \n
        rte = ||Tt - Te|| \n
    '''
    return np.linalg.norm(Tt - Te)

def patch_coordinates(origin, patch_size):
    o_h, o_w = origin
    patch_h, patch_w = patch_size
    y = np.arange(o_h, o_h + patch_h)
    x = np.arange(o_w, o_w + patch_w)
    g = np.meshgrid(y, x) # TODO
    img = np.vstack(map(np.ravel, g)).T
    return img.astype(float)

@torch.no_grad()
def get_pose(pc, K, origin, patch_size, dist_thres=5): # 5 pixels
    pc = pc.cpu().detach().numpy().astype(float)
    pc = pc[:, :3] # keep only xyz
    img = patch_coordinates(origin, patch_size)
    print(pc.shape, pc.dtype)
    print(img.shape, img.dtype)

    img = img[:1024]

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

def get_errors(pc, origin, patch_size, K, Rt, Tt, dist_thres=5):
    Re, Te = get_pose(pc, K, origin, patch_size, dist_thres)
    return RRE(Rt, Re), RTE(Tt, Te)
