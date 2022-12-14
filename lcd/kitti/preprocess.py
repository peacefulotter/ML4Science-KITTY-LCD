
import os
import numpy as np
import open3d as o3d
from typing import Literal
from loguru import logger

from pointcloud import downsample_neighbors
import calib
import plots

class KittiPreprocess:

    KITTI_DATA_FOLDER = 'kitti_data'
    DATASET_TYPES = Literal["all", "train", "test", "debug"]
    SEQ_LISTS = {
        "all": list(range(11)),
        "train": list(range(9)), 
        "test": [9, 10], 
        "debug": [0]
    }

    def __init__(self, root, mode: DATASET_TYPES, img_width=64, img_height=64, num_pc=4096, min_pc=128):
        if mode != "debug":
            logger.disable(__name__)

        self.root = root
        self.mode = mode
        self.img_w = img_width
        self.img_h = img_height
        self.num_pc = num_pc
        self.min_pc = min_pc
        
        self.dataset = self.make_kitti_dataset()
        self.calib = calib.import_calib(root)


    def get_dataset_folder(self, seq, name):
        return os.path.join(self.root, 'sequences', '%02d' % seq, name)

    def make_kitti_dataset(self):
        dataset = []
        seq_list = KittiPreprocess.SEQ_LISTS[self.mode]
        logger.debug(f'Loading {seq_list} sequences')
        for seq_i in seq_list:
            img2_folder = self.get_dataset_folder(seq_i, 'img_P2')
            img3_folder = self.get_dataset_folder(seq_i, 'img_P3')
            K2_folder = self.get_dataset_folder(seq_i, 'K_P2')
            K3_folder = self.get_dataset_folder(seq_i, 'K_P3')
            pc_folder = self.get_dataset_folder(seq_i, 'pc_npy_with_normal')
            samples = round(len(os.listdir(img2_folder)))
            for img_i in range(samples):
                dataset.append((img2_folder, pc_folder, K2_folder, seq_i, img_i, 'P2'))
                dataset.append((img3_folder, pc_folder, K3_folder, seq_i, img_i, 'P3'))
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

    def crop_img(self, img, max_point_height = np.inf):
        w, h = self.img_w, self.img_h
        dx = np.random.randint(0, min(img.shape[1] - w, max_point_height))
        dy = np.random.randint(0, img.shape[0] - h)
        return img[dy: dy + h, dx: dx + w, :], dx, dy

    def display_points_in_image(self, depth_mask, in_frame_mask, pc):
        total_mask = self.combine_masks(depth_mask, in_frame_mask)
        colors = np.zeros(pc.shape)
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
        # colors = intensity / max_intensity   # colors *

        # colors=np.zeros((pc.shape[0],3))
        # colors[:,0:1]= intensity /max_intensity

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

    def get_cropped_img(self, center, img):
        '''
        params:
            center = (1, 3) - point that can be projected on the image
            img = (H, W, 3) array of rgb values
        returns:
            cropped_img = (self.img_h, self.img_w)
        '''
        x, y = center[0], center[1]
        o_x = max(0, x - self.img_w / 2)
        o_y = max(0, y - self.img_h / 2)
        return img[o_y : o_y + self.img_h, o_x : o_x + self.img_w]

    @staticmethod
    def resolve_img_folder(root, seq_i, img_i):
        return os.path.join(
            root,
            KittiPreprocess.KITTI_DATA_FOLDER, # main data folder
            str(seq_i), # sequence index
            str(img_i), # image index
        )

    @staticmethod
    def resolve_data_path(img_folder, i):
        return os.path.join(
            img_folder,
            '%06d.npz' % i # pointcloud (neighbors) index for this image index
        )

    @staticmethod
    def load_data(img_folder, i):
        '''
        img_folder: using KittiPreprocess.resolve_img_folder
        i: ith sample
        '''
        path = KittiPreprocess.resolve_data_path(img_folder, i)
        data = np.load(path)
        return data['pc'], data['img']

    def save_data(self, img_folder, i, pc, img):
        path = KittiPreprocess.resolve_data_path(img_folder, i)
        logger.info(f'Storing at {path}  - rgb_pc: {pc.shape}, img: {img.shape}')
        np.savez(path, pc=pc, img=img)


    def get_samples(self, seq_i, img_i):
        img_folder = KittiPreprocess.resolve_img_folder(self.root, seq_i, img_i)
        samples = 0
        if os.path.exists(img_folder):
            samples = round(len(os.listdir(img_folder))) 
        return img_folder, samples

    def store_result(self, seq_i, img_i, pc_in_frame, colors, centers, img, neighbors_indices):

        assert centers.shape[0] == neighbors_indices.shape[0]

        # nb of samples already preprocessed for this sequence and this image
        img_folder, samples = self.get_samples(seq_i, img_i) 

        # root = os.path.join(self.root, 'lcd', 'kitti')
        for i, indices in enumerate(neighbors_indices):
            center = centers[i]
            neighbors_pc = pc_in_frame[indices]
            neighbors_rgb = colors[indices]
            neighbors_rgb_pc = np.c_[ neighbors_pc, neighbors_rgb ]
            cropped_img = self.get_cropped_img(center, img)

            self.make_nested_folders(root, [KittiPreprocess.KITTI_DATA_FOLDER, str(seq_i), str(img_i)])
            self.save_data(img_folder, samples + i + 1, neighbors_rgb_pc, cropped_img)

    def make_nested_folders(self, root, folders):
        for folder in folders:
            root = os.path.join(root, folder)
            print(root)
            if not (os.path.exists(root)):
                os.mkdir(root)

    def __getitem__(self, index):
        (
            img_folder, pc_folder, K_folder, # data folders
            seq_i, img_i, # ith sequence and ith image
            key, # key=(P2 or P3)
        ) = self.dataset[index]
        img, pc, intensity, sn, K = self.load_item(img_folder, pc_folder, K_folder, img_i)

        logger.info(f'{seq_i}, {img_i}, {key}')

        # Take random part of the image of size (img_w, img_h)
        # img, dx, dy = self.crop_img(img)

        # Project point cloud to image
        Pi = self.calib[seq_i][key] # (3, 4)
        Tr = self.calib[seq_i]['Tr'] # (4, 4)
        P_Tr = np.matmul(Pi, Tr)

        pts_ext = np.r_[ pc, np.ones((1,pc.shape[1])) ]
        pts_cam = (P_Tr @ pts_ext)[:3, :].T

        # cam_x = pts_cam_T[:, 0]
        # cam_y = pts_cam_T[:, 1]
        depth = pts_cam[:, 2]

        # discard the points behind the camera (of negative depth) -- these points get flipped during the z_projection
        depth_mask = ~(depth < 0.1)
        pts_front_cam = pts_cam[depth_mask]
        intensity_front_cam = intensity.T[depth_mask]
        sn_front_cam = sn.T[depth_mask]

        # TODO better K
        K = np.eye(3) * 1 / 100
        # TODO: playing around with this value changes the pc
        # K = camera_matrix_scaling(K, 1/4)  # the 1/4 is the number I saw in CorrI2P
        # K = camera_matrix_cropping(K, dx=dx, dy=dy)

        pts_front_cam = (K @ pts_front_cam.T).T
        # max_point_height = max(pts_front_cam[:, 2]) # heighest z val in point cloud projected to camera

        def z_projection(pts):
            z = pts[:, 2:3]
            return pts / z, z

        pts_on_camera_plane, z = z_projection(pts_front_cam)

        # take the points falling inside the image
        in_image_mask = (
            (pts_on_camera_plane[:, 0] >= 0) &
            (pts_on_camera_plane[:, 0] < img.shape[1]) &
            (pts_on_camera_plane[:, 1] >= 0) &
            (pts_on_camera_plane[:, 1] < img.shape[0])
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
        pc_in_frame = pc.T[total_mask]

        # TODO: delete? (retries when no points)
        if pc_in_frame.shape[0] == 0:
            logger.warn('Not enough points projected, retrying')
            return self.__getitem__(index)

        # Voxel Downsample pointcloud
        ds_pc_space, ds_colors, ds_sn = self.voxel_down_sample(pc_in_frame, intensity_in_frame, sn_in_frame, colors)
        # ds_pc_space, ds_colors = self.downsample_np(ds_pc_space, ds_colors)
        logger.info(f'[Downsample] original: {pc_in_frame.shape}, downsampled: {ds_pc_space.shape}')

        # Find neighbors for each point
        # TODO: debug only
        ds_pc_space = ds_pc_space[:10]
        neighbors_indices, centers = downsample_neighbors(ds_pc_space, pc_in_frame, self.min_pc)
        logger.info(f'[Neighbors] original: {pc_in_frame.shape}, {neighbors_indices.shape}')

        self.store_result(seq_i, img_i, pc_in_frame, colors, centers, img, neighbors_indices)


if __name__ == '__main__':

    w, h = 64, 64
    num_pc = pow(2,  10)
    min_pc = 32
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    print('root:', root)
    dataset = KittiPreprocess(
        root=root,
        mode='debug', # TODO: change this to "all"
        img_width=w,
        img_height=h,
        num_pc=num_pc,
        min_pc=min_pc
    )

    for i in range(15):
        dataset[i]

    pc, img = KittiPreprocess.load_data(seq_i=0, img_i=10, i=5)
    plots.plot_rgb_pc(pc)