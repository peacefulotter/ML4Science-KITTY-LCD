
import os
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data

from calib import import_calib

# TODO: ImageFolder dataset
class KittyDataset(data.Dataset):
    def __init__(self, root, mode, num_pc, img_width, img_height, *args, **kwargs):
        super(KittyDataset, self).__init__(*args, **kwargs)
        self.root = root
        self.mode = mode
        self.num_pc = num_pc
        self.img_width = img_width
        self.img_height = img_height
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
            pc_folder = self.get_dataset_folder(seq, 'pc_npy_with_normal')
            K2_folder = self.get_dataset_folder(seq, 'K_P2')
            K3_folder = self.get_dataset_folder(seq, 'K_P3')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                K2_folder, seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                K3_folder, seq, i, 'P3', sample_num))
        return dataset

    def load_npy(self, folder, seq_i):
        return np.load(os.path.join(folder, '%06d.npy' % seq_i))

    def load_item(self, img_folder, pc_folder, seq_i):
        img = self.load_npy(img_folder, seq_i)
        data = self.load_npy(pc_folder, seq_i)
        pc = data[:3, :]
        return img, pc

    def __len__(self):
        return len(self.dataset) 


    def bounding_box(self, points, min_x, max_x, min_y, max_y):
        """ Compute a bounding_box filter on the given points

        Parameters
        ----------                        
        points: (n,2) array
            The array containing all the points's coordinates. Expected format:
                array([
                    [x1,y1],
                    ...,
                    [xn,yn]])

        min_i, max_i: float
            The bounding box limits for each coordinate. If some limits are missing,
            the default values are -infinite for the min_i and infinite for the max_i.

        Returns
        -------
        bb_filter : boolean array
            The boolean mask indicating wherever a point should be keeped or not.
            The size of the boolean mask will be the same as the number of given points.

        """
        bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] < max_x)
        bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] < max_y)
        return np.logical_and(bound_x, bound_y)

    def __getitem__(self, index):
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img, pc = self.load_item(img_folder, pc_folder, seq_i)

        # Project point cloud to image
        Pi = self.calib[seq][key] # (3, 4)
        Tr = self.calib[seq]['Tr'] # (4, 4)
        X = np.r_[ pc, np.ones((1, pc.shape[1])) ]   # (X, Y, Z, 1), shape=(4, N)
        x = Pi @ Tr @ X # (3, N) where the last row is not used
        x = np.delete(x, (0), axis=0) # (2, N)
       
        # TODO: take random subpart of img
        # of size (img_width, img_height)

        # Mask the pc to only keep the points projected onto the image
        in_img_mask = self.bounding_box(x.T, 0, self.img_width, 0, self.img_height) # (M, )
        print("Original nb points =", pc.shape[1])
        print("Keeping nb points =", in_img_mask[in_img_mask == True].shape[0])
        x_in_img = x[:, in_img_mask] # (2, M)
        pc_in_img = pc[:, in_img_mask] # (3, M)
        
        # get RGB color of each point in the pc_in_img
        pc_img_mask = np.round(x_in_img).astype(int) # (2, M) (int)
        colors = img[ pc_img_mask[0, :], pc_img_mask[1, :] ] # (M, 3) RGB per point

        # concatenate pc with rgb color
        pc = np.r_[ pc_in_img, colors.T / 255 ]  
        return img, pc


if __name__ == '__main__':
    idx = 0
    w, h = 320, 320
    dataset = KittyDataset(root="./", mode='train', num_pc=0, img_width=w, img_height=h)
    img, pc_colored = dataset[idx]
    pc, colors = np.vsplit(pc_colored, 2)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc.T)
    pointcloud.colors = o3d.utility.Vector3dVector(colors.T)
    o3d.visualization.draw_geometries([pointcloud])