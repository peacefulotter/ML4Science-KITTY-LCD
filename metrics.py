import cv2
import numpy as np

from scipy.spatial.transform import Rotation


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

def get_P_pred(pc_sel, img_xy_pc, K, *args):
    _, R, t, _ = cv2.solvePnPRansac(
        pc_sel.T, img_xy_pc.T, K, useExtrinsicGuess=False,
        iterationsCount=500, reprojectionError=args.dist_thres, 
        flags=cv2.SOLVEPNP_EPNP, distCoeffs=None
    )
    R, _ = cv2.Rodrigues(R)
    
    P_pred = np.eye(4)
    P_pred[:3, :3] = R
    P_pred[:3, 3:] = t

    return P_pred

# Relative Translational Error (RTE)
# Relative Rotational Error (RRE)
# K: cam intrinsic?
def relative_error(pc, img_feature, pc_feature, img_score, pc_score, P_gr, K, *args):
    pc_sel, img_xy_pc = get_img_xy_pc(pc, img_feature, pc_feature, img_score, pc_score, *args)
    P_pred = get_P_pred(pc_sel, img_xy_pc, K, *args)
    
    P_diff = np.linalg.inv(P_pred) @ P_gr
    rte = np.linalg.norm(P_diff[0:3, 3])
    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    rre = np.sum(np.abs(R_diff.as_euler('xyz', degrees=True)))
    
    return rte, rre