import os
import torch.utils.data as data

from .preprocess import KittiPreprocess

class KittiDataset(data.Dataset):
    '''
    This is the dataset class to use the preprocessed data 
    from KittiPreprocess. 

    This class is pretty straightforward as KittiPreprocess is the one
    that does all the work, i.e. this class basically maps the index (given in __get_item__) 
    to a tuple (sequence index, image index, sample index) and loads the preprocessed
    files at `root / KITTI_DATA_FOLDER / sequence_index / image_index / sample_index`. 
    '''

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
        '''
        Reads and saves the file structure.
        For each sequence folder in the kitti data folder:
            For each image folder in the sequence folder:
                number of samples = number of files
        Ignores other types of files such as calibs files.
        '''
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


    # In the end not used because of the many projection issues
    # but might be useful for the lab later 
    # def import_calibs(self):
    #     calibs = [{} for i in range(len(self.seq_list))]
    #     base_folder = os.path.join(self.root, KittiPreprocess.KITTI_DATA_FOLDER)
    #     for seq_i in self.seq_list:
    #         path = os.path.join(base_folder, str(seq_i), 'calib.npz')
    #         data = np.load(path)
    #         calibs[seq_i] = {'P2': data['P2'], 'P3': data['P3']}
    #     return calibs

    def map_index(self, idx):
        '''
        Maps an index 'idx' to a tuple (seq_i, img_i, sample_i)
        using the file structure built during the class initialization 
        (see build_dataset)
        '''
        samples = 0
        for seq_i, seq_samples in self.samples.items():
            for img_i, img_samples in seq_samples.items():
                if samples + img_samples > idx:
                    return seq_i, img_i, idx - samples
                samples += img_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        '''
        1. Map the given index to a tuple (seq_i, img_i, sample_i) using the file structure
        built during class initialization
        2. Load the preprocessed npz file at `root / KITTI_DATA_FOLDER / seq_i / img_i / sample_i`. 
        3. Return the RGB colored pointcloud and patch

        Returns:
        - (n, 6) pc: RGB colored pointcloud
        - (patch_h, patch_w, 3): RGB patch image, patch_h and patch_w are parameters defined during preprocessing
        '''
        seq_i, img_i, sample_i = self.map_index(index)
        pc, patch, _ = KittiPreprocess.load_data(self.root, seq_i, img_i, sample_i)
        return pc, patch