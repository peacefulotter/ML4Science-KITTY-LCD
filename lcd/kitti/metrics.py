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

@torch.no_grad()
def get_pose(pc, img, K, dist_thres=5): # 5 pixels
    pc = pc.cpu().detach().numpy().astype(float)
    img = img.cpu().detach().numpy().astype(float)
    K = K.cpu().detach().numpy().astype(float)
    print(pc.dtype, img.dtype, K.dtype)
    _, R, t, _ = cv2.solvePnPRansac(
        pc, img, K, 
        # useExtrinsicGuess=False,
        # iterationsCount=500, 
        reprojectionError=dist_thres, 
        flags=cv2.SOLVEPNP_EPNP, 
        distCoeffs=None
    )
    R, _ = cv2.Rodrigues(R) # Converts rotation vector to matrix
    return R, t

def get_errors(pc, img, K, Rt, Tt, dist_thres=5):
    Re, Te = get_pose(pc, img, K, dist_thres)
    return RRE(Rt, Re), RTE(Tt, Te)
