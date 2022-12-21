import os
import torch.utils.data as data

from .preprocess import KittiPreprocess

class KittiDataset(data.Dataset):
    def __init__(self, root, mode, *args, **kwargs):
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

        print(pc.shape)

        return pc, img