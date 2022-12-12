
import os
import numpy as np
import open3d as o3d
import torch.utils.data as data

from pointcloud import downsample_neighbors
import calib
import plots

# TODO: ImageFolder dataset
class KittiPreprocess:
    def __init__(self, root, mode, img_width=64, img_height=64, num_pc=4096, min_pc=128):
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

    def store_result(self, seq_i, img_i, pc_in_frame, colors, img, centers, neighbors_indices):

        assert centers.shape[0] == neighbors_indices.shape[0]

        root = os.path.join(self.root, 'lcd', 'kitti')
        for i, indices in enumerate(neighbors_indices):
            center = centers[i]
            center_pc = pc_in_frame[indices]
            center_rgb = colors[indices]
            center_rgb_pc = np.c_[ center_pc, center_rgb ]
            cropped_img = self.get_cropped_img(center, img)

            self.make_nested_folders(root, ['kitti_data', str(seq_i), str(img_i)])
            path = os.path.join(root, 'kitti_data', str(seq_i), str(img_i), f'{i}.npz')
            print('Storing at', path, 'rgb_pc:', center_rgb_pc.shape, ', img:', cropped_img.shape)
            plots.compare_pc_with_colors(
                pc_in_frame, colors,
                center_pc, 'red',
                np.array([center]), 'green'
            )
            np.savez(path, center=center, pc=center_rgb_pc, img=cropped_img)

    def make_nested_folders(self, root, folders):
        for folder in folders:
            root = os.path.join(root, folder)
            if not (os.path.exists(root)):
                os.mkdir(root)



    def __getitem__(self, index):
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img, pc, intensity, sn, K = self.load_item(img_folder, pc_folder, K_folder, seq_i)

        # Take random part of the image of size (img_w, img_h)
        # img, dx, dy = self.crop_img(img)

        # Project point cloud to image
        Pi = self.calib[seq][key] # (3, 4)
        Tr = self.calib[seq]['Tr'] # (4, 4)
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

        # TODO: delete (retries when no points)
        if pc_in_frame.shape[0] == 0:
            print('Not enough points projected, retrying')
            return self.__getitem__(index)

        # Voxel Downsample pointcloud
        ds_pc_space, ds_colors, ds_sn = self.voxel_down_sample(pc_in_frame, intensity_in_frame, sn_in_frame, colors)
        # ds_pc_space, ds_colors = self.downsample_np(ds_pc_space, ds_colors)
        print("downsample: ", pc_in_frame.shape, ds_pc_space.shape)

        # Find neighbors for each point
        # TODO: debug only
        ds_pc_space = ds_pc_space[:10]
        neighbors_indices, centers = downsample_neighbors(ds_pc_space, pc_in_frame, self.min_pc)
        print("neighbors: ", pc_in_frame.shape, neighbors_indices.shape, centers.shape)
        print(neighbors_indices)

        self.store_result(seq, seq_i, pc_in_frame, colors, img, centers, neighbors_indices)


if __name__ == '__main__':

    w, h = 64, 64
    num_pc = pow(2,  10)
    min_pc = 32
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    dataset = KittiPreprocess(
        root=root,
        mode='train',
        img_width=w,
        img_height=h,
        num_pc=num_pc,
        min_pc=min_pc
    )

    for i in range(10):
        dataset[i]