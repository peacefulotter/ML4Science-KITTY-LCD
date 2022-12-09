
import os
import numpy as np
import torch.utils.data as data

from .calib import import_calib
from .plots import plot_pc

import open3d as o3d

# TODO: ImageFolder dataset
class KittiDataset(data.Dataset):
    def __init__(self, root, mode, num_pc=4096, img_width=64, img_height=64, *args, **kwargs):
        super(KittiDataset, self).__init__(*args, **kwargs)
        self.root = root
        self.mode = mode
        self.num_pc = num_pc
        self.img_w = img_width
        self.img_h = img_height
        self.dataset = self.make_kitti_dataset()
        self.calib = import_calib(root)

    def get_dataset_folder(self, seq, name):
        return os.path.join(self.root, 'sequences', '%02d' % seq, name)

    def make_kitti_dataset(self):
        dataset = []

        # TODO: replace with list(range(9))
        seq_list = [0] if self.mode == 'train' else [9, 10]

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = self.get_dataset_folder(seq, 'img_P2')
            img3_folder = self.get_dataset_folder(seq, 'img_P3')
            K2_folder = self.get_dataset_folder(seq, 'K_P2')
            K3_folder = self.get_dataset_folder(seq, 'K_P3')
            pc_folder = self.get_dataset_folder(seq, 'pc_npy_with_normal')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder, K2_folder, seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder, K3_folder, seq, i, 'P3', sample_num))

        return dataset

    def load_npy(self, folder, seq_i):
        return np.load(os.path.join(folder, '%06d.npy' % seq_i))

    def load_item(self, img_folder, pc_folder, K_folder, seq_i):
        img = self.load_npy(img_folder, seq_i)
        data = self.load_npy(pc_folder, seq_i)
        K = self.load_npy(K_folder, seq_i)
        pc = data[0:3, :]
        intensity = data[3:4, :]
        sn = data[4:, :] # surface normals
        return img, pc, intensity, sn, K

    def __len__(self):
        return len(self.dataset)

    def crop_img(self, img):
        w, h = self.img_w, self.img_h
        dx = np.random.randint(0, img.shape[1] - w)
        dy = np.random.randint(0, img.shape[0] - h)
        return img[dy: dy + h, dx: dx + w, :], dx, dy

    def display_points_in_image(self, depth_mask, in_frame_mask, pc):
        total_mask = self.combine_masks(depth_mask, in_frame_mask)
        colors = np.zeros(pc.shape)
        print(colors.shape)
        colors[1, total_mask] = 190/255 # highlight selected points in green
        colors[2, ~total_mask] = 120/255 # highlight unselected points in blue
        plots.plot_pc(pc.T, colors.T)

    #Combine sequential masks: where the second mask is used after the first one
    def combine_masks(self, depth_mask, in_frame_mask):
        mask = np.zeros(depth_mask.shape)
        idx_in_frame = 0
        for idx_depth, depth in enumerate(depth_mask):
            if depth:
                mask[idx_depth] = in_frame_mask[idx_in_frame]
                idx_in_frame += 1
        return mask.astype(bool)

    def voxel_down_sample(self, pc, intensity, sn, colors, voxel_grid_size=.1):
        # TODO: use intensity?
        max_intensity = np.max(intensity)
        # colors = colors * intensity / max_intensity   

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(sn)

        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_size)
        down_pcd_points = np.asarray(down_pcd.points)
        down_pcd_colors = np.asarray(down_pcd.colors)
        down_pcd_sn = np.asarray(down_pcd.normals)

        # down_pcd_colors *= max_intensity

        return down_pcd_points, down_pcd_colors, down_pcd_sn

    def downsample_np(self, pc, colors):
        nb_points = pc.shape[0]
        if nb_points >= self.num_pc:
            choice_idx = np.random.choice(nb_points, self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(nb_points))
            while nb_points + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(nb_points))), axis=0)
            random_idx = np.random.choice(nb_points, self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        return pc[choice_idx], colors[choice_idx]

    def __getitem__(self, index):
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img, pc, intensity, sn, K = self.load_item(img_folder, pc_folder, K_folder, seq_i)

        # Take random part of the image of size (img_w, img_h)
        img, dx, dy = self.crop_img(img)

        # Project point cloud to image
        Pi = self.calib[seq][key] # (3, 4)
        Tr = self.calib[seq]['Tr'] # (4, 4)

        Pi_ext = np.row_stack((Pi, [0, 0, 0, 1]))
        Tr_ext = np.row_stack((Tr, [0, 0, 0, 1]))
        P_Tr = np.matmul(Pi_ext, Tr_ext)

        pts_ext = np.row_stack((pc, np.ones((pc.shape[1],))))
        pts_cam = (P_Tr @ pts_ext)[:3, :]
        pts_cam_T = pts_cam.T

        # cam_x = pts_cam_T[:, 0]
        # cam_y = pts_cam_T[:, 1]
        depth = pts_cam_T[:, 2]

        # discard the points behind the camera (of negative depth) -- these points get flip during the z_projection
        depth_mask = ~(depth < 0.1)
        pts_front_cam = pts_cam_T[depth_mask]
        intensity_front_cam = intensity.T[depth_mask]
        sn_front_cam = sn.T[depth_mask]

        def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
            K_crop = np.copy(K)
            K_crop[0, 2] -= dx
            K_crop[1, 2] -= dy
            return K_crop

        def camera_matrix_scaling(K: np.ndarray, s: float):
            K_scale = s * K
            K_scale[2, 2] = 1
            return K_scale

        # K = np.eye(3)
        # TODO: playing around with this value changes the pc
        K = camera_matrix_scaling(K, 1 / 1000)  # the 1/4 is the number I saw in CorrI2P
        K = camera_matrix_cropping(K, dx=dx, dy=dy)

        pts_front_cam = (K @ pts_front_cam.T).T

        def z_projection(pts):
            z = pts[:, 2:3]
            return pts / z, z

        pts_on_camera_plane, z = z_projection(pts_front_cam)

        # take the points falling inside the image
        in_image_mask = (
            (pts_on_camera_plane[:, 0] >= 0) &
            (pts_on_camera_plane[:, 0] < self.img_w) &
            (pts_on_camera_plane[:, 1] >= 0) &
            (pts_on_camera_plane[:, 1] < self.img_h)
        )
        pts_in_frame = pts_on_camera_plane[in_image_mask]
        intensity_in_frame = intensity_front_cam[in_image_mask]
        sn_in_frame = sn_front_cam[in_image_mask]

        # TODO: add some noise to the color_mask
        color_mask = np.floor(pts_in_frame).astype(int)
        colors = img[ color_mask[:, 1], color_mask[:, 0] ] / 255 # (M, 3) RGB per point
        #pts_in_frame = pts_in_frame*z[in_image_mask]

        # Get the pointcloud in its original space
        total_mask = self.combine_masks(depth_mask, in_image_mask)
        pc_space = pc.T[total_mask]

        # TODO: delete (retries when no points)
        if pc_space.shape[0] == 0:
            return self.__getitem__(index)

        # Downsample pointcloud
        pc_space, colors, sn = self.voxel_down_sample(pc_space, intensity_in_frame, sn_in_frame, colors)
        pc_space, colors = self.downsample_np(pc_space, colors)
        pc_space = np.c_[ pc_space, colors ]

        return pc_space, img


if __name__ == '__main__':
    # idx = 50
    # w, h = 256, 128
    w, h = 64, 64
    for p in range(5):
        num_pc = pow(2, p + 12)
        dataset = KittiDataset(
            root="./", 
            mode='train', 
            num_pc=num_pc, 
            img_width=w, 
            img_height=h
        )
        pc, img = dataset[0]
        pc, colors = np.hsplit(pc, 2)
        plots.plot_pc(pc, colors)