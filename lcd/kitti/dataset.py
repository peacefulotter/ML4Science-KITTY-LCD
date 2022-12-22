import os
import numpy as np
import torch.utils.data as data

from lcd.kitti.preprocess import KittiPreprocess
import lcd.kitti.metrics

class KittiDataset(data.Dataset):
    def __init__(self, root, mode: KittiPreprocess.DATASET_TYPES, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.seq_list = KittiPreprocess.SEQ_LISTS[mode]
        # self.calibs = self.import_calibs()
        self.total_samples = 0
        self.samples = {} # seq_i -> img_i -> nb of samples
        self.build_dataset()
        print('--------- KittiDataset init Done ---------')
        print(' > total samples:', self.total_samples)
        print(' > samples structure:', self.samples)

    def build_dataset(self):
        base_folder = os.path.join(self.root, KittiPreprocess.KITTI_DATA_FOLDER)
        for seq_i in self.seq_list:
            self.samples[seq_i] = {}
            seq_path = os.path.join(base_folder, str(seq_i))
            img_folders = os.listdir(seq_path)
            nb_img = 0
            for img_folder in img_folders:
                img_path = os.path.join(seq_path, img_folder)
                if not os.path.isdir(img_path):
                    continue
                img_samples = len(os.listdir(img_path))
                self.total_samples += img_samples
                self.samples[seq_i][img_folder] = img_samples
                nb_img += 1

    # TODO: (focus on projection first), in case we really need to save preprocessed calib files and 
    # import them in the dataset 
    # def import_calibs(self):
    #     calibs = [{} for i in range(len(self.seq_list))]
    #     base_folder = os.path.join(self.root, KittiPreprocess.KITTI_DATA_FOLDER)
    #     for seq_i in self.seq_list:
    #         path = os.path.join(base_folder, str(seq_i), 'calib.npz')
    #         data = np.load(path)
    #         calibs[seq_i] = {'P2': data['P2'], 'P3': data['P3']}
    #     return calibs

    def map_index(self, idx):
        samples = 0
        for seq_i, seq_samples in self.samples.items():
            for img_i, img_samples in seq_samples.items():
                if samples + img_samples > idx:
                    return seq_i, img_i, idx - samples
                samples += img_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        seq_i, img_i, sample = self.map_index(index)
        img_folder = KittiPreprocess.resolve_img_folder(self.root, seq_i, img_i)
        pc, img, Pi = KittiPreprocess.load_data(img_folder, sample)
        return pc, img


if __name__ == '__main__':

    import sys
    sys.path.append('../')
    from models.patchnet import PatchNetAutoencoder
    from models.pointnet import PointNetAutoencoder

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    dataset = KittiDataset(root=root, mode='debug')

    """pc, img = dataset[199]
    pc, colors = np.hsplit(pc, 2)
    plots.plot_pc(pc, colors)"""

    loader = data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )
    device = 'cpu'
    patchnet = PatchNetAutoencoder(256, True)
    pointnet = PointNetAutoencoder(256,6,6,True)
    patchnet.to(device)
    pointnet.to(device)

    for i, batch in enumerate(loader):
        x = [x.to(device).float() for x in batch]

        # plots.plot_rgb_pc(batch[0][0])

        pred_pcs, point_descriptors = pointnet(x[0])
        pred_imgs, patch_descriptors = patchnet(x[1])
        # Ks = x[2]

        print("input batch", x[0].shape, x[1].shape)
        print("pointnet output", pred_pcs.shape, point_descriptors.shape)
        print("pathnet output", pred_imgs.shape, patch_descriptors.shape)

        # for pred_rgb_pc, pred_img in zip(pred_pcs, pred_imgs):
        #     pred_pc, colors = np.hsplit(pred_rgb_pc, 2)
        #     pred_R, pred_t = metrics.get_pose(pred_pc, pred_img, K)
        #     print("pred_pose")
        #     print(pred_R.shape, pred_t.shape)
        #     print(pred_R, pred_t)