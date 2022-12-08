# %%
import IPython
if (ipython := IPython.get_ipython()) is not None:
	ipython.run_line_magic("load_ext", "autoreload")
	ipython.run_line_magic("autoreload", "2")

# import open3d as o3d
import torch
import numpy as np
from kitti import dataset
from corri2p.dataset import kitti_pc_img_dataset

from matplotlib import pyplot as plt
from PIL import Image


#%%
ds = dataset.KittiDataset(
	root=".", 
	mode="train", 
	num_pc=1024)

# ds[0]  # in the render generated from this call, 
  		 # I can't see any points. Did I miss something?
#%%
# below is an working example of projection;
# I directly load a point cloud and an image
# the peculiar thing is the matrix K, being unused here
# CorrI2P uses K for cropping and scaling, but
# the K loaded from the `K_P2` dir (line 42 this file)
# is too large -- this causes the point cloud blows up
# I use an identity matrix to substitute, and
# it works pretty fine. But the cropping and scale 
# is not handled by the code automatically, is an todo
# but at least, all the transformation works

sample_id = "000009"
point_data = np.load(f"./sequences/00/pc_npy_with_normal/{sample_id}.npy")  
points = point_data[0:3, :]
intensity = point_data[3, :]
normals = point_data[4:7, :]

w, h = 300, 128

img = np.load(f"./sequences/00/img_P2/{sample_id}.npy")
print(img.shape)
K = np.load(f"./sequences/00/K_P2/{sample_id}.npy")

P2 = ds.calib[0]["P2"]
Tr = ds.calib[0]["Tr"] 
P2_ext = np.row_stack((P2, [0, 0, 0, 1]))
Tr_ext = np.row_stack((Tr, [0, 0, 0, 1]))
P_Tr = np.matmul(P2_ext, Tr_ext)
# P_Tr = Tr_ext
points_ext = np.row_stack((points, np.ones((points.shape[1],))))
points_cam = (P_Tr @ points_ext)[:3, :]

pts_cam_T = points_cam.T

cam_x = pts_cam_T[:, 0]
cam_y = pts_cam_T[:, 1]
depth = pts_cam_T[:, 2]

# discard the points behind the camera (of negative depth) -- these points get flip during the z_projection
pts_front_cam = pts_cam_T[~(depth < 0.1)]

def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
	K_crop = np.copy(K)
	K_crop[0, 2] -= dx
	K_crop[1, 2] -= dy
	return K_crop

def camera_matrix_scaling(K: np.ndarray, s: float):
	K_scale = s * K
	K_scale[2, 2] = 1
	return K_scale

img_crop_dx = int((img.shape[1] - w) / 2)
img_crop_dy = int((img.shape[0] - h) / 2)
# img = img[img_crop_dy: img_crop_dy + h, img_crop_dx: img_crop_dx + w, :]
# img = img[img_crop_dy: img_crop_dy + h, :w]
# K = camera_matrix_scaling(K, 1 / 707 / 4)
# K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

K = np.eye(3)
K = camera_matrix_scaling(K, 1 / 4)  # the 1/4 is the number I saw in CorrI2P

pts_front_cam = (K @ pts_front_cam.T).T

def z_projection(pts):
	z = pts[:, 2:3]
	return pts / z

pts_on_camera_plane = z_projection(pts_front_cam)

print(pts_on_camera_plane.shape)

in_image_mask = (  # take the points falling inside the image
	  (pts_on_camera_plane[:, 0] >= 0) 
	& (pts_on_camera_plane[:, 0] < w) 
	& (pts_on_camera_plane[:, 1] >= 0) 
	& (pts_on_camera_plane[:, 1] < h) 
)

print(np.count_nonzero(in_image_mask))
pts_in_frame = pts_on_camera_plane[in_image_mask]
plt.subplot(2, 1, 1)
plt.scatter(pts_in_frame[:, 0], pts_in_frame[:, 1], c=pts_front_cam[in_image_mask, 2], s=0.05)
plt.axis("scaled")
plt.subplot(2, 1, 2)
plt.imshow(img, origin="lower")  # origin="lower" for the same coordinate system as the scatter plot
plt.axis("scaled")
plt.show()

# %%
# implement the colorization of the point cloud
# choose the RGB from the image using the coords of the points

# %%
