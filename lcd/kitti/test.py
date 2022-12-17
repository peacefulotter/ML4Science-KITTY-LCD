import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import KittiPreprocess
from poses import import_poses, get_pos

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    preprocess = KittiPreprocess(root, mode='debug')
    calibs = preprocess.calibs
    calib = calibs[0]

    idx = 0
    img_folder, pc_folder, seq_i, img_i, cam_i = preprocess.dataset[idx]
    img, pc, intensity, sn = preprocess.load_item(img_folder, pc_folder, img_i)

    poses = import_poses(root, seq_i)
    gt = poses[img_i]

    print("ground truth pose", gt)

    print("pc shape ", pc.shape)
    # P2 (3 x 4) for left eye
    Pi_rect = calib[f'P{cam_i}']

    # Decomposing a projection matrix with OpenCV
    k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(Pi_rect)
    t1 = t1 / t1[3]
    t1 = t1[:3]
    print('Pose Matrix:')
    print(Pi_rect)
    print('Intrinsic Matrix:')
    print(k1)
    print('Rotation Matrix:')
    print(r1)
    print('Translation Vector:')
    print(t1.round(4))

    # Tr = calib['T2']
    # P2 = P2_rect # np.vstack((P2_rect, [0, 0, 0, 1]))
    # pc = np.vstack((pc, np.ones((1, pc.shape[1]))))

    # Rt = np.hstack((r1, t1))
    # print('Rt matrix\n', Rt)
    # pc_c = Rt @ pc
    # Z_c = pc_c[2]

    # cam = (k1 @ pc_c) / Z_c

    """
    # CORRI2P
    from calib import get_p, get_tr
    Pi = get_p(Pi_rect)
    Tr = get_tr(calib['Tr'])
    P_Tr = Pi @ Tr
    K = np.load(os.path.join(root, 'sequences', '%02u' % seq_i, 'K_P2', '%06d.npy' % seq_i))
    K = 0.25 * K
    K[2, 2] = 1
    # rotation * pc + translation
    pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]
    pc_ = np.dot(K, pc)
    pc_mask = np.zeros((1, np.shape(pc)[1]), dtype=np.float32)
    pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
    cam = np.floor(pc_[0:2, :])
    """
    print(pc.shape)
    pc = np.vstack((pc, np.ones((1, pc.shape[1]))))
    K = np.eye(3, 4)
    K[:3, :3] = calib['K2']
    cam = K @ calib['T2'] @ pc

    cam = cam / cam[2]
    # cam = P2 @ Tr @ pc 
    # cam = K @ Rt @ pc
    print(cam.shape)
    print(cam)

    # depth_from_cam = cam[2]
    # cam = cam / depth_from_cam
    # print(cam)

    u, v, z  = cam
    IMG_H, IMG_W, _ = img.shape
    in_frame = np.logical_and(
        np.logical_and(u >= 0, u < IMG_W), 
        np.logical_and(v >= 0, v < IMG_H)
    )
    print(in_frame[in_frame == True].shape, in_frame[in_frame == False].shape,)
    cam = cam[:, in_frame]

    u, v, z  = cam
    z = pc[2, :]
    z = z[in_frame]
    # z  = Z_c[in_frame]

    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    plt.axis([0,IMG_W,IMG_H,0])
    plt.scatter([u],[v],c=[z],cmap='plasma_r',s=2)
    plt.imshow(img)
    plt.show()