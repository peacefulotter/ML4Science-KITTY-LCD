
import os
import cv2
import numpy as np
from plots import plot_pc
import matplotlib.pyplot as plt

from PIL import Image

from calib import read_calib_file, parse_calib_file, get_p


def combine_masks(depth_mask, in_frame_mask):
    mask = np.zeros(depth_mask.shape)
    idx_in_frame = 0
    for idx_depth, depth in enumerate(depth_mask):
        if depth:
            mask[idx_depth] = in_frame_mask[idx_in_frame]
            idx_in_frame += 1
    return mask.astype(bool)

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    root = os.path.join(root, 'raw_kitti')

    img_path = os.path.join(root, 'sequences', '%02d' % 0, 'image_2', '000000.png')
    calib_path = os.path.join(root, 'sequences', '%02d' % 0, 'calib.txt')
    pc_path = os.path.join(root, 'sequences',  '%02d' % 0, 'velodyne', '000000.bin')

    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    pc = pc[:, :3].T  # exclude luminance

    img = np.array(Image.open(img_path))
    img_h, img_w, _ = img.shape

    calib = read_calib_file(root, 0)
    calib = parse_calib_file(calib)


    Pi = calib['P2']
    # Pi = get_p(Pi)

    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(Pi)
    t = t / t[3]
    t = t[:3]
    Rt = np.hstack((r, t))

    for i in range(pc.shape[1]):
        point = pc[:, i]
        point_homogeneous = np.append(point, 1)
        point_2d = np.matmul(Pi, point_homogeneous)
        point_2d = point_2d[:2] / point_2d[2]
        point_2d = np.floor(point_2d).astype(int)
        u, v = point_2d
        if u >= 0 and u < img_h and v >= 0 and v < img_w:
            print(i)
            print(point)
            print(point_2d)
        if u >= 0 and u < img_w and v >= 0 and v < img_h:
            print(i)
            print(point)
            print(point_2d)

    pc_ext = np.r_[ pc, np.ones((1,pc.shape[1])) ]
    # pts_cam = (Pi @ pc_ext)[:3, :].T
    pts_cam = (k @ Rt @ pc_ext)[:3, :].T

    depth = pts_cam[:, 2]
    depth_mask = ~(depth < 0.1)
    pts_front_cam = pts_cam[depth_mask]

    z = pts_front_cam[:, 2:3]
    pts_on_camera_plane = pts_front_cam / z

    # take the points falling inside the image
    in_image_mask = (
        (pts_on_camera_plane[:, 0] >= 0) &
        (pts_on_camera_plane[:, 0] < img_h) &
        (pts_on_camera_plane[:, 1] >= 0) &
        (pts_on_camera_plane[:, 1] < img_w)
    )
    pts_in_frame = pts_on_camera_plane[in_image_mask]

    # Get RGB for each point on the image
    color_mask = np.floor(pts_in_frame).astype(int)
    projected_colors = img[ color_mask[:, 1], color_mask[:, 0] ] / 255 # (M, 3) RGB per point
    
    # Get the pointcloud back using the masks indices
    total_mask = combine_masks(depth_mask, in_image_mask)
    pc_in_frame = pc.T[total_mask]
    
    print(pc_in_frame.shape)
    # plot_pc(pc_in_frame)

    plt.figure()
    plt.imshow(img)
    plt.scatter(pts_in_frame[:, 0], pts_in_frame[:, 1], c=z[in_image_mask], cmap='plasma_r', marker=".", s=5)
    plt.colorbar()
    plt.show()