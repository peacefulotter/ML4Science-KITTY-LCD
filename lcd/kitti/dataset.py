import os
import numpy as np
import torch.utils.data as data

from preprocess import KittiPreprocess
import metrics

class KittiDataset(data.Dataset):
    def __init__(self, root, mode: KittiPreprocess.DATASET_TYPES, *args, **kwargs):
        super(KittiDataset, self).__init__(*args, **kwargs)
        self.root = root
        self.seq_list = KittiPreprocess.SEQ_LISTS[mode]
        self.calibs = self.import_calibs()

    def import_calibs(self):
        calibs = [{} for i in range(len(self.seq_list))]
        base_folder = os.path.join(self.root, KittiPreprocess.KITTI_DATA_FOLDER)
        for seq_i in self.seq_list:
            path = os.path.join(base_folder, str(seq_i), 'calib.npz')
            data = np.load(path)
            calibs[seq_i] = {'P2': data['P2'], 'P3': data['P3']}
        return calibs

    def __len__(self):
        # TODO: find way to compute length
        return 2 # len(self.dataset)

    def __getitem__(self, index):
        # TODO: find mapping (index) -> (seq_i, img_i, sample)
        seq_i = 0
        img_i = 0
        sample = 18 if index == 0 else 10
        
        img_folder = KittiPreprocess.resolve_img_folder(self.root, seq_i, img_i)
        pc, img, P_i = KittiPreprocess.load_data(img_folder, sample)
        K = self.calibs[seq_i][P_i]

        return pc, img, K


if __name__ == '__main__':

    import sys
    sys.path.append('../')
    from models.patchnet import PatchNetAutoencoder
    from models.pointnet import PointNetAutoencoder

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    dataset = KittiDataset(root=root, mode='train')

    """pc, img = dataset[199]
    pc, colors = np.hsplit(pc, 2)
    plots.plot_pc(pc, colors)"""

    loader = data.DataLoader(
        dataset,
        batch_size=2,
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
        Ks = x[2]

        print("input batch", x[0].shape, x[1].shape)
        print("pointnet output", pred_pcs.shape, point_descriptors.shape)
        print("pathnet output", pred_imgs.shape, patch_descriptors.shape)

        for pred_rgb_pc, pred_img, K in zip(pred_pcs, pred_img, Ks):
            pred_pc, colors = np.hsplit(pred_rgb_pc, 2)
            pred_R, pred_t = metrics.get_pose(pred_pc, pred_img, K)
            print("pred_pose")
            print(pred_R.shape, pred_t.shape)
            print(pred_R, pred_t)