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


# TODO: see evaluate_odometry.cpp
# from KITTI devkit


# define args.img_thres
#        args.pc_thres
#        args.dist_thres

def get_img_xy_pc(pc, img_feature, pc_feature, img_score, pc_score, *args):
    img_feature_shape = img_feature.shape 
    img_score_shape = img_score.shape

    img_x = np.linspace(0, img_feature_shape[-1] - 1, img_feature_shape[-1])
    img_x = img_x.reshape(1, -1).repeat(img_feature_shape[-2], 0)

    img_y = np.linspace(0, img_feature_shape[-2] - 1, img_feature_shape[-2])
    img_y = img_y.reshape(-1, 1).repeat(img_feature_shape[-1], 1)

    img_x = img_x.reshape(1, img_score_shape[-2], img_score_shape[-1])
    img_y = img_y.reshape(1, img_score_shape[-2], img_score_shape[-1])
    img_xy = np.concatenate((img_x,img_y), axis=0)
    
    img_xy_flatten = img_xy.reshape(2,-1)
    img_feature_flatten = img_feature.reshape(np.shape(img_feature)[0],-1)
    img_score_flatten = img_score.squeeze().reshape(-1)
    img_index = img_score_flatten > args.img_thres
    img_xy_flatten_sel = img_xy_flatten[:,img_index]
    img_feature_flatten_sel = img_feature_flatten[:,img_index]
    img_feature = np.expand_dims(img_feature_flatten_sel,axis=1)

    pc_index = pc_score.squeeze() > args.pc_thres
    pc_sel = pc[:, pc_index]
    pc_feature_sel=pc_feature[:, pc_index]
    pc_feature_sel = np.expand_dims(pc_feature_sel,axis=2)

    dist = 1 - np.sum(pc_feature_sel * img_feature, axis=0)
    sel_index = np.argmin(dist,axis=1)
    img_xy_pc = img_xy_flatten_sel[:,sel_index]

    return pc_sel, img_xy_pc

@torch.no_grad()
def get_pose(pc, img, K, dist_thres=1):
    # TODO: why pnpransac
    # TODO: what are those parameters
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
    
    # pose = np.eye(4)
    # pose[:3, :3] = R
    # pose[:3, 3:] = t
    return R, t

# Relative Translational Error (RTE)
# Relative Rotational Error (RRE)
# K: cam intrinsic?
def relative_error(pc, img_feature, pc_feature, img_score, pc_score, P_gr, K, *args):
    pc_sel, img_xy_pc = get_img_xy_pc(pc, img_feature, pc_feature, img_score, pc_score, *args)
    P_pred = get_pose(pc_sel, img_xy_pc, K, args.dist_thres)
    
    P_diff = np.linalg.inv(P_pred) @ P_gr
    rte = np.linalg.norm(P_diff[0:3, 3])
    r_diff = P_diff[0:3, 0:3]
    R_diff = R.from_matrix(r_diff)
    rre = np.sum(np.abs(R_diff.as_euler('xyz', degrees=True)))
    
    return rte, rre