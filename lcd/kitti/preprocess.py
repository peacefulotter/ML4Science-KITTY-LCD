
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

    def __init__(self, root, mode: DATASET_TYPES, patch_w=64, patch_h=64, num_pc=4096, min_pc=128):
        if mode != "debug":
            logger.disable(__name__)

        self.root = root
        self.mode = mode
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.num_pc = num_pc
        self.min_pc = min_pc
        
        self.dataset = self.make_kitti_dataset()
        self.calib = calib.import_calib(root)


    def get_dataset_folder(self, seq, name):
        return os.path.join(self.root, 'sequences', '%02d' % seq, name)

    def make_kitti_dataset(self):
        '''
        Returns:
            Array where each element follows the following format:
            img_folder, pc_folder, K_folder, # data folders
            seq_i, img_i, # ith sequence and ith image
            key, # key=(P2 or P3)
        '''
        dataset = []
        seq_list = KittiPreprocess.SEQ_LISTS[self.mode]
        logger.info(f'Loading {seq_list} sequences')
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

    # TODO: delete or move to plots
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

    # voxel_grid_size not to lose too much info up
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

    def get_cropped_img(self, center, img):
        '''
        params:
            center = (1, 3) - point that can be projected on the image
            img = (H, W, 3) array of rgb values
        returns:
            cropped_img = (self.patch_h, self.patch_w)
        '''
        w, h = center[0], center[1]
        img_w, img_h = img.shape[1], img.shape[0]
        o_h = min(max(0, int(h - (self.patch_h / 2))), img_h - self.patch_h)
        o_w = min(max(0, int(w - (self.patch_w / 2))), img_w - self.patch_w)
        # TODO: neighbors radius should be similar to cropped img w and h ??
        return img[o_h : o_h + self.patch_h, o_w : o_w + self.patch_w]

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
    def load_data(img_folder, sample):
        '''
        img_folder: using KittiPreprocess.resolve_img_folder
        sample: ith sample
        '''
        path = KittiPreprocess.resolve_data_path(img_folder, sample)
        data = np.load(path)
        return data['pc'], data['img'], data['Pi'].item()

    # Storage issue /!\  
    def save_data(self, img_folder, i, pc, img, Pi):
        path = KittiPreprocess.resolve_data_path(img_folder, i)
        logger.info(f'Storing at {path}  - rgb_pc: {pc.shape}, img: {img.shape}')
        np.savez(path, pc=pc, img=img, Pi=Pi)


    def get_samples(self, img_folder):
        samples = 0
        if os.path.exists(img_folder):
            samples = round(len(os.listdir(img_folder))) 
        return samples

    def store_result(self, seq_i, img_i, pc_in_frame, colors, centers_2D, img, neighbors_indices, Pi):

        assert centers_2D.shape[0] == neighbors_indices.shape[0]
        
        img_folder = KittiPreprocess.resolve_img_folder(self.root, seq_i, img_i)
        # nb of samples already preprocessed for this sequence and this image
        samples = self.get_samples(img_folder) 

        for i, indices in enumerate(neighbors_indices):
            center = centers_2D[i]
            neighbors_pc = pc_in_frame[indices]
            neighbors_rgb = colors[indices]
            neighbors_rgb_pc = np.c_[ neighbors_pc, neighbors_rgb ]
            cropped_img = self.get_cropped_img(center, img)
            self.make_nested_folders([str(seq_i), str(img_i)])
            self.save_data(img_folder, samples + i, neighbors_rgb_pc, cropped_img, Pi)

    def make_nested_folders(self, folders):
        folders.insert(0, KittiPreprocess.KITTI_DATA_FOLDER)
        root = self.root
        for folder in folders:
            root = os.path.join(root, folder)
            if not (os.path.exists(root)):
                os.mkdir(root)
        return root

    def save_calib_files(self):
        for seq_i, seq_calib in enumerate(self.calib):
            path = self.make_nested_folders([str(seq_i)])
            res = {}
            for key, mat in seq_calib.items():
                if key == 'P2' or key == 'P3':
                    res[key] = mat
            path = path + '/calib'
            logger.info(f"Saving calib file {seq_i} to {path}")
            np.savez(path, P2=res['P2'], P3=res['P3'])


    def get_pc_in_frame(self, pc, img, seq_i, key, K):

        # Project point cloud to image
        Pi = self.calib[seq_i][key] # (3, 4)

        # Pi[:3, :3] = Pi[:3, :3] * 2

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
        
        # pts_front_cam = (Pi[:3, :3] @ pc).T #  (Pi[:3, :3] @  [depth_mask]
        # pts_front_cam[:, 0] += img.shape[1] / 2
        # pts_front_cam[:, 1] += img.shape[0] / 2
        # pts_front_cam = (K @ pts_front_cam.T).T
        # max_point_height = max(pts_front_cam[:, 2]) # heighest z val in point cloud projected to camera

        def z_projection(pts):
            z = pts[:, 2:3]
            return pts / z, z

        pts_on_camera_plane, z = z_projection(pts_front_cam)
        pts_on_camera_plane[: 1] = img.shape[0] - pts_on_camera_plane[: 1]

        # take the points falling inside the image
        in_image_mask = (
            (pts_on_camera_plane[:, 0] >= 0) &
            (pts_on_camera_plane[:, 0] < img.shape[1]) &
            (pts_on_camera_plane[:, 1] >= 0) &
            (pts_on_camera_plane[:, 1] < img.shape[0])
        )
        pts_in_frame = pts_on_camera_plane[in_image_mask]

        offset = -img.shape[0] / 6
        pts_in_frame_offset = pts_in_frame.copy()
        pts_in_frame_offset[:, 1] = pts_in_frame_offset[:, 1] + offset

        color_mask = np.floor(pts_in_frame_offset).astype(int)
        projected_colors = img[ color_mask[:, 1], color_mask[:, 0] ] / 255 # (M, 3) RGB per point
        total_mask = self.combine_masks(depth_mask, in_image_mask)
        plot_pc = pc.T[total_mask]
        plot_pc[:, 1] += offset
        # plots.plot_pc(plot_pc, projected_colors)

        import matplotlib.pyplot as plt
        # # ax[0].imshow(img)
        plt.imshow(img)
        plt.scatter(pts_in_frame_offset[:, 0], pts_in_frame_offset[:, 1], c=z[in_image_mask], cmap='plasma_r', marker=".", s=5)
        plt.colorbar()
        plt.show()

        return pts_in_frame, depth_mask, in_image_mask

    def full_projection(self, pc, intensity, sn, img, seq_i, key, K):
        
        pts_in_frame, depth_mask, in_image_mask = self.get_pc_in_frame(pc, img, seq_i, key, K)

        # TODO: add some noise to the color_mask
        color_mask = np.floor(pts_in_frame).astype(int)
        projected_colors = img[ color_mask[:, 1], color_mask[:, 0] ] / 255 # (M, 3) RGB per point
        #pts_in_frame = pts_in_frame*z[in_image_mask]

        # Get the pointcloud in its original space
        total_mask = self.combine_masks(depth_mask, in_image_mask)
        pc_in_frame = pc.T[total_mask]
        intensity_in_frame = intensity.T[total_mask]
        sn_in_frame = sn.T[total_mask]

        return pc_in_frame, intensity_in_frame, sn_in_frame, projected_colors

    def remove_center_outliers(self, centers_2D, img):
        u, v, z = centers_2D
        h, w, _ = img.shape
        min_w = self.patch_w / 2
        min_h = self.patch_h / 2
        u_in = np.logical_and(u >= min_w, u <= w - min_w)
        v_in = np.logical_and(v >= min_h, v <= h - min_h)
        inliers_mask = np.logical_and(u_in, v_in)
        centers_2D = centers_2D.T[inliers_mask]
        return centers_2D, inliers_mask


    def __getitem__(self, index):
        img_folder, pc_folder, K_folder, seq_i, img_i, key = self.dataset[index]

        img, pc, intensity, sn, K = self.load_item(img_folder, pc_folder, K_folder, img_i)
        pc_in_frame, intensity_in_frame, sn_in_frame, projected_colors = self.full_projection(pc, intensity, sn, img, seq_i, key, K)

        # TODO: delete? (retries when no points)
        if pc_in_frame.shape[0] == 0:
            logger.warn('Not enough points projected, retrying')
            return self.__getitem__(index)

        # plots.plot_pc(pc_in_frame, projected_colors)

        # Voxel Downsample pointcloud
        ds_pc, ds_colors, ds_sn = self.voxel_down_sample(pc_in_frame, intensity_in_frame, sn_in_frame, projected_colors)
        # ds_pc_space, ds_colors = self.downsample_np(ds_pc, ds_colors)
        logger.info(f'[Downsample] original: {pc_in_frame.shape}, downsampled: {ds_pc.shape}')

        # Find neighbors for each point
        neighbors_indices, centers_3D  = downsample_neighbors(ds_pc, pc_in_frame, self.min_pc)
        logger.info(f'[Neighbors] original: {pc_in_frame.shape}, {neighbors_indices.shape}')
        
        # Project the 3D neighbourhoods center 
        centers_2D, center_depth_mask, in_image_center_mask = self.get_pc_in_frame(centers_3D.T, img, seq_i, key, K)
        centers_2D = np.floor(centers_2D).astype(int)
        centers_2D = centers_2D.T

        # Again, centers from the voxel down sample might fall outside the image bounds = need to filter
        center_total_mask = self.combine_masks(center_depth_mask, in_image_center_mask)
        neighbors_indices = neighbors_indices[center_total_mask]
        centers_3D = centers_3D[center_total_mask]

        # Remove points that fall outside of the img bounds
        centers_2D, inliers_mask = self.remove_center_outliers(centers_2D, img)

        neighbors_indices = neighbors_indices[inliers_mask]
        centers_3D = centers_3D[inliers_mask]
       
        self.store_result(seq_i, img_i, pc_in_frame, projected_colors, centers_2D, img, neighbors_indices, Pi=key)


if __name__ == '__main__':

    w, h = 64, 64
    num_pc = pow(2,  10)
    min_pc = 64
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    logger.info(f'Loading from root: {root}')
    preprocess = KittiPreprocess(
        root=root,
        mode='debug', # TODO: change this to "all"
        patch_w=w,
        patch_h=h,
        num_pc=num_pc,
        min_pc=min_pc
    )   

    # TODO: test.py did not work 
    # TODO: offset fails for some image
    # TODO: + different offset for P3

    start_idx = 0 # TODO: replace with 70
    for i in range(start_idx, 100, 2):
        img_folder, pc_folder, K_folder, seq_i, img_i, key = preprocess.dataset[i]
        img, pc, intensity, sn, K = preprocess.load_item(img_folder, pc_folder, K_folder, img_i)
        preprocess.get_pc_in_frame(pc, img, seq_i, key, K) 

    # Save preprocessed calib files    
    # preprocess.save_calib_files()

    # Used to preprocess the kitti data and save it to the KittiPreprocess.KITTI_DATA_FOLDER
    # preprocess[1]
    # for i in range(15):
    #    preprocess[i]

    seq_i = 0
    img_i = 2
    samples = 20
    img_folder = preprocess.resolve_img_folder(root, seq_i, img_i)
    for sample in range(samples):
        pc, img, Pi = preprocess.load_data(img_folder, sample)
        logger.success(f'sample: {sample} {pc.shape}, {img.shape}, {Pi}')
        # plots.plot_rgb_pc(pc)