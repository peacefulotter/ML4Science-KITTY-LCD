import os
import json
import torch
import numpy as np
import torch.utils.data as data

from lcd.models import *
from lcd.losses import *

from lcd.kitti.eval_dataset import KittiEvalDataset
from lcd.kitti.descriptors import find_descriptors_correspondence

from lcd.kitti.metrics import get_estimated_pose, RRE, RTE

@torch.no_grad()
def main():

    config = "./config_kitti.json"
    logdir = "./logs/LCD"

    args = json.load(open(config))
    print(args)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    fname = os.path.join(logdir, "config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    device = 'cuda' if args["device"] == 'cuda' and torch.cuda.is_available() else 'cpu'

    dataset = KittiEvalDataset(
        root=args['root'],
        patch_w=args['patch_w'],
        patch_h=args['patch_h'],
        min_pc=args['min_pc']
    )
    loader = data.DataLoader(
        dataset,
        batch_size=2, # args["batch_size"],
        # num_workers=1, # args["num_workers"],
        # pin_memory=True,
        collate_fn=dataset.collate,
        shuffle=True,
    )

    patchnet = PatchNetAutoencoder(args["embedding_size"], args["normalize"])
    pointnet = PointNetAutoencoder(
        args["embedding_size"],
        args["input_channels"],
        args["output_channels"],
        args["normalize"],
    )
    patchnet.to(device)
    pointnet.to(device)

    rtes = []
    rres = []

    for i, batch in enumerate(loader):
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)

        print(f'[{i}] Forwading to model')
        '''
        Forward the evaluation batch containing all the possible img patches
        of size (img_h, img_w) and "all" the neighbourhoods (1024 pointcloud points per point)
        '''
        x = [x.to(device).float() for x in batch[:2]]
        _, z0 = pointnet(x[0])
        _, z1 = patchnet(x[1])

        print(f'[{i}] Establishing descriptor correspondences')
        '''
        Establish the descriptor correspondences i.e. find the nearest neighbor 
        from z0 to z1. 
        '''
        correspondences = find_descriptors_correspondence(z0, z1)
        for i, correspondence in enumerate(correspondences):
            '''
            For all found correspondences, get the camera intrinsic parameters (K),
            get the ground truth rotation matrix and translation vector from the ground
            truth pose and finally compute the RRE and RTE using PnPRansac and Rodrigues.
            '''
            pc = batch[0][i]
            origin = batch[2][correspondence]
            seq_i, img_i, cam_i = batch[3][i]
            patch_size = (args['patch_h'], args['patch_w'])

            K = dataset.intrinsic_params(seq_i, cam_i)
            Rt, Tt = dataset.get_extracted_pose(seq_i, img_i)
            Re, Te = get_estimated_pose(pc, K, origin, patch_size)
            rre, rte = RRE(Rt, Re), RTE(Tt, Te)
            
            rres.append(rre)
            rtes.append(rte)
        
        print('--------------')
        print('RRE', np.array(rres).mean(), ' - ', np.array(rres).std())
        print('RTE', np.array(rtes).mean(), ' - ', np.array(rtes).std())
        
    print(' > Done')


if __name__ == '__main__':
    main()