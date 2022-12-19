
import os
import numpy as np
from plots import plot_pc
import matplotlib.pyplot as plt

from PIL import Image

from calib import read_calib_file, parse_calib_file, get_p, import_cam_to_cam, import_velo_to_cam

def proj_matrix(calib, cam_i, Rt):
    """
    Compute the projection matrix from LiDAR to Img
    Parameters:
        filename: filename of the calib file
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Img
    """
    R_rect = calib[f'R_0{cam_i}'].reshape((3, 3))
    P_rect = calib[f'P_rect_0{cam_i}'].reshape((3, 4))
    velo_to_cam = np.vstack([Rt,np.array([0,0,0,1])])

    R_cam2rect = np.hstack([R_rect, np.array([[0],[0],[0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0,0,0,1])])
    
    return P_rect @ R_cam2rect @ velo_to_cam

def project(root, img, img_i, cam_i):
    pc_path = os.path.join(root, "velodyne_points", 'data', '%010u.bin' % img_i)
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    pc = pc[:, :3].T 
    
    Rt = import_velo_to_cam(root)
    cam_to_cam = import_cam_to_cam(root)

    proj_mat = proj_matrix(cam_to_cam, cam_i, Rt)
    pc_hg = np.r_[ pc, np.ones((1,pc.shape[1])) ]
    pts_2d_hg = proj_mat @ pc_hg # Pi @ Rt_hg @ pc_hg
    depth = pts_2d_hg[2]

    depth_mask = ~(depth < 0.1)
    pts_front_cam = pts_2d_hg[:, depth_mask]

    z = pts_front_cam[2:3]
    pts_2d = pts_front_cam / z

    # take the points falling inside the image
    pixels = np.floor(pts_2d).astype(int)
    u, v, _ = pixels
    img_h, img_w, _ = img.shape
    in_image_mask = (
        (u >= 0) & (u < img_w) &
        (v >= 0) & (v < img_h)
    )
    pixels_in_frame = pixels[:, in_image_mask]
    z = z.T[in_image_mask]

    return pixels_in_frame, z

def import_img(root, img_i, cam_i):
    img_path = os.path.join(root, f"image_0{cam_i}", 'data', '%010u.png' % img_i)
    img = np.array(Image.open(img_path))
    return img

def plot(img, pixels, z, img_i, cam_i):
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    u, v, _ = pixels
    plt.scatter(u, v, c=z, cmap='plasma_r', marker=".", s=5)
    plt.colorbar()
    plt.title(f'Img: {img_i}, cam: {cam_i}')
    plt.show()

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    root = os.path.join(root, 'raw')

    seq_i = 0
    for img_i in range(20):
        cam_i = 2
        img = import_img(root, img_i, cam_i)
        pixels, z = project(root, img, img_i, cam_i)
        plot(img, pixels, z, img_i, cam_i)

        cam_i = 3
        img = import_img(root, img_i, cam_i)
        pixels, z = project(root, img, img_i, cam_i)
        plot(img, pixels, z, img_i, cam_i)
    
    