import os
import numpy as np
# import open3d as o3d

from .pointcloud import downsample_neighbors
from .projection import project, project_kitti
from .calib import import_calibs

class KittiPreprocess:
    '''
    This is the preprocessing class for using the KITTI dataset on
    the LCD input format.

    See the __get_item__ function for more informations on how the preprocessing works.
    '''

    KITTI_DATA_FOLDER = 'kitti_data'
    MODES = ['all', 'train', 'test', 'debug']
    SEQ_LISTS = {
        "all": list(range(11)),
        "train": list(range(9)),
        "test": [9, 10],
        "debug": [0]
    }

    def __init__(self, root, mode, patch_w=64, patch_h=64, num_pc=1024, min_pc=32):
        '''
        patch_w: patch image width 
        patch_h: patch image height
        num_pc: number of points in each pointcloud
        min_pc: mininum number of points in a neighborhood, if less discard
        '''
        if mode not in KittiPreprocess.MODES:
            raise Exception('mode parameter should be one of ' + str(KittiPreprocess.MODES))

        self.root = root
        self.mode = mode
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.num_pc = num_pc
        self.min_pc = min_pc
        self.seq_list = KittiPreprocess.SEQ_LISTS[mode]

        self.dataset = self.make_kitti_dataset()
        self.calibs = import_calibs(root, self.seq_list)

        print('--------- KittiPreprocess  init Done ---------')


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
        print(f" Dataset consists of {self.seq_list} sequences ")
        for seq_i in self.seq_list:
            img2_folder = self.get_dataset_folder(seq_i, 'img_P2')
            img3_folder = self.get_dataset_folder(seq_i, 'img_P3')
            K2_folder = self.get_dataset_folder(seq_i, 'K_P2')
            K3_folder = self.get_dataset_folder(seq_i, 'K_P3')
            pc_folder = self.get_dataset_folder(seq_i, 'pc_npy_with_normal')
            samples = round(len(os.listdir(img2_folder)))
            for img_i in range(samples):
                # img_folder, pc_folder, sequence index, img index, camera index
                dataset.append((img2_folder, pc_folder, seq_i, img_i, 2))
                dataset.append((img3_folder, pc_folder, seq_i, img_i, 3))
        return dataset

    def load_npy(self, folder, img_i):
        return np.load(os.path.join(folder, '%06d.npy' % img_i))

    def load_item(self, img_folder, pc_folder, img_i):
        img = self.load_npy(img_folder, img_i)
        data = self.load_npy(pc_folder, img_i)
        # K = self.load_npy(K_folder, seq_i)
        pc = data[0:3, :]
        intensity = data[3:4, :]
        sn = data[4:, :] # surface normals
        return img, pc, intensity, sn

    def __len__(self):
        return len(self.dataset) 

    # voxel_grid_size not to lose too much info up
    def voxel_down_sample(self, pc, sn, colors, voxel_grid_size=.1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(sn)

        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_size)
        down_pcd_points = np.asarray(down_pcd.points)
        down_pcd_colors = np.asarray(down_pcd.colors)
        down_pcd_sn = np.asarray(down_pcd.normals)

        return down_pcd_points, down_pcd_colors, down_pcd_sn

    def get_cropped_img(self, center, img):
        '''
        Params:
            - (1, 3) center: Point that can be projected onto the image
            - (H, W, 3) img: RGB image
        Returns:
            - (self.patch_h, self.patch_w) cropped img 
        '''
        w, h = center[0], center[1]
        img_w, img_h = img.shape[1], img.shape[0]
        o_h = min(max(0, int(h - (self.patch_h / 2))), img_h - self.patch_h)
        o_w = min(max(0, int(w - (self.patch_w / 2))), img_w - self.patch_w)
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
    def load_data(root, seq_i, img_i, sample_i):
        '''
        sample: ith sample
        '''
        img_folder = KittiPreprocess.resolve_img_folder(root, seq_i, img_i)
        path = KittiPreprocess.resolve_data_path(img_folder, sample_i)
        data = np.load(path)
        return data['pc'], data['img'], data['Pi'].item()

    def save_data(self, img_folder, i, pc, img, cam_i):
        '''
        Save the data for a single sample (pair of num_pc points and (patch_h, patch_w) image).

        Each "sample" file contains the following:
        - (n, 6) pc: The RGB colored pointcloud containing only points 
        that can be projected onto the image frame and lie in the image bounds
        - (patch_h, patch_w, 3) img: RGB patch image
        - Pi: ith camera (TODO: refactor name) 
        '''
        if self.mode == "all":
            # Save data on SCITAS in the scratch folder
            seq_dir, img_dir = os.path.split(img_folder)[-2:]
            print("output: ", end=" ")
            img_folder = os.path.join(os.path.expandvars("$SCRATCH"), "kitti", seq_dir, img_dir)
            print(img_folder)
            os.makedirs(img_folder, exist_ok=True)
        elif self.mode == "debug":
            pass
        else:
            raise ValueError(f"mode {self.mode} unsupported")

        path = KittiPreprocess.resolve_data_path(img_folder, i)
        np.savez(path, pc=pc, img=img, Pi=cam_i)

    def get_samples(self, img_folder):
        '''
        Get the number of samples already present in the given img_folder
        '''
        samples = 0
        if os.path.exists(img_folder):
            samples = round(len(os.listdir(img_folder)))
        return samples

    def store_result(self, seq_i, img_i, pc_in_frame, colors, centers_2D, img, neighbors_indices, cam_i):
        '''
        Save the preprocessed data into numerous "sample" npz files.
        The file structure is as following:
        - KITTI_DATA_FOLDER / seq_i / img_i / XXXXXXXX.npz 
        See KittiPreprocess.save_data to know the format for a single file 
        '''
        assert centers_2D.shape[0] == neighbors_indices.shape[0]

        img_folder = KittiPreprocess.resolve_img_folder(self.root, seq_i, img_i)
        # nb of samples already preprocessed for this sequence and this image
        samples = self.get_samples(img_folder)

        print(f' > Storing in {img_folder} {len(neighbors_indices)} samples')

        for i, indices in enumerate(neighbors_indices):
            '''
            For each list of indices in the neighbors_indices list:
            1. Get the (n, 3) points and (n, 3) rgb values
            2. Concatenate them to have an RGB color per point into an (n, 6) shaped array
            3. Get the patch of size (patch_h, patch_w) where 'center' lies at the center of the patch
            4. Create the nested folders where the sample will be saved
            5. Save the colored pointcloud along with the patch (see save_data)  
            '''
            center = centers_2D[i]
            neighbors_pc = pc_in_frame[indices]
            neighbors_rgb = colors[indices]
            neighbors_rgb_pc = np.c_[ neighbors_pc, neighbors_rgb ]
            cropped_img = self.get_cropped_img(center, img)
            self.make_nested_folders([str(seq_i), str(img_i)])
            self.save_data(img_folder, samples + i, neighbors_rgb_pc, cropped_img, cam_i)

    def make_nested_folders(self, folders):
        '''
        Create a series of nested folders starting from the kitti data folder
        '''
        folders.insert(0, KittiPreprocess.KITTI_DATA_FOLDER)
        root = self.root
        for folder in folders:
            root = os.path.join(root, folder)
            if not (os.path.exists(root)):
                os.mkdir(root)
        return root

    # Might be useful for the lab later
    # def save_calib_files(self):
    #     for seq_i, seq_calib in enumerate(self.calibs):
    #         path = self.make_nested_folders([str(seq_i)])
    #         res = {}
    #         for key, mat in seq_calib.items():
    #             if key == 'P2' or key == 'P3':
    #                 res[key] = mat
    #         path = path + '/calib'
    #         print(f" > Saving calib file {seq_i} to {path}")
    #         np.savez(path, P2=res['P2'], P3=res['P3'])

    def remove_center_outliers(self, centers_2D, img):
        '''
        Remove points that lie outside the given image bounds
        '''
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
        '''
        Main function: preprocessed the image at index "index" in the built dataset.
        (see make_kitti_dataset).

        1. Loads the pointcloud and image as numpy arrays
        2. Project the pointcloud on the camera frame and retrieve the points
        that project onto the image
        3. Voxel down sample the n projected points
        4. Use the downsampled points as centers for computing the neighbourhoods of size num_pc
        (1024 for LCD). Discard the neighbourhoods of size less then min_pc and down or upsample the 
        neighbourhoods depending on their size (upsample if size in [min_pc, num_c] and downsample if
        size > num_pc)
        5. Project the centers onto the image and retrieve a patch of (patch_h, patch_w) around each center.
        6. Discard the centers where we can't obtain a patch such that the center is actually projected at the center of the patch (i.e.)
        the projected center is too close to the image bounds, discard it.
        7. Save the num_pc points neighbourhoods along with their colors and the patches (see store_result) 
        '''
        img_folder, pc_folder, seq_i, img_i, cam_i = self.dataset[index]
        print(f'--------- Preprocessing {index}  -  seq_i: {seq_i}, img_i: {img_i}, cam_i: {cam_i} ---------')

        # 1. Load the img and pc
        img, pc, intensity, sn = self.load_item(img_folder, pc_folder, img_i)
        
        # 2. Project the pc in the camera frame 
        pc_in_frame, _, sn_in_frame, projected_colors = project_kitti(
            self.calibs, pc, intensity, sn, img, seq_i, cam_i
        )

        # 3. Voxel Downsample pointcloud
        ds_pc, _, _ = self.voxel_down_sample(pc_in_frame, sn_in_frame, projected_colors)
        print(f' > [Downsample] original: {pc_in_frame.shape}, downsampled: {ds_pc.shape}')

        # 4. Find neighbors for each point
        neighbors_indices, centers_3D  = downsample_neighbors(ds_pc, pc_in_frame, self.min_pc)
        print(f' > [Neighbors] original: {pc_in_frame.shape}, {neighbors_indices.shape}')

        # 5. Project the 3D neighbourhoods center
        centers_2D, _, _, center_total_mask = project(centers_3D.T, img, seq_i, cam_i)
        centers_2D = np.floor(centers_2D).astype(int)
        centers_2D = centers_2D.T

        '''
        Should not happen because voxel down sample takes the average
        # Again, centers from the voxel down sample might fall outside the image bounds = need to filter
        neighbors_indices = neighbors_indices[center_total_mask]
        centers_3D = centers_3D[center_total_mask]
        '''

        # 6. Remove points that fall outside of the img bounds
        centers_2D, inliers_mask = self.remove_center_outliers(centers_2D, img)
        neighbors_indices = neighbors_indices[inliers_mask]
        centers_3D = centers_3D[inliers_mask]
        
        # 7. Save the result to sample files
        self.store_result(seq_i, img_i, pc_in_frame, projected_colors, centers_2D, img, neighbors_indices, cam_i)



if __name__ == '__main__':

    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("--mode", choices=["debug", "all"], default="debug")
    args = argparser.parse_args()

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    print(f' > Loading from root: {root}')
    preprocess = KittiPreprocess(
        root=root,
        mode=args.mode,  # depends on program argument, default to "arg"
    )

    # Save preprocessed calib files
    # TODO: should we really? first figure out the projection
    # preprocess.save_calib_files()

    # Used to preprocess the kitti data and save it to the KittiPreprocess.KITTI_DATA_FOLDER
    for i in range(0, len(preprocess), 2):
        preprocess[i]

    seq_i = 0
    img_i = 2
    samples = 20
    for sample_i in range(samples):
        pc, img, Pi = preprocess.load_data(root, seq_i, img_i, sample_i)
        print(f' > Success, sample: {sample_i} {pc.shape}, {img.shape}, {Pi}')
        # plots.plot_rgb_pc(pc)