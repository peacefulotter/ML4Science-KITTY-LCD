import os
import json
import torch
import numpy as np
import torch.utils.data as data

from lcd.models import *
from lcd.losses import *

from lcd.kitti.eval_dataset import KittiEvalDataset
from lcd.descriptors import find_descriptors_correspondence

from lcd.kitti.metrics import get_errors

@torch.no_grad()
def main():

    config = "./config_kitti.json"
    logdir = "./logs/LCD"

    args = json.load(open(config))

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    fname = os.path.join(logdir, "config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    device = 'cuda' if args["device"] == 'cuda' and torch.cuda.is_available() else 'cpu'

    dataset = KittiEvalDataset(args["root"])
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

    for i, batch in enumerate(loader):
        print(f'[{i}] Forwading to model')
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        x = [x.to(device).float() for x in batch[:2]]
        print(x[0].shape, x[1].shape)
        pcs, z0 = pointnet(x[0])
        patches, z1 = patchnet(x[1])

        print(z0.shape, z1.shape)
        print(f'[{i}] Establishing descriptor correspondences')
        correspondences = find_descriptors_correspondence(z0, z1)
        for i, correspondence in enumerate(correspondences):
            pc = pcs[i]
            patch = patches[correspondence]
            seq_i, img_i, cam_i = batch[2][i]
            print(seq_i, img_i, cam_i)
            Rt, Tt = dataset.get_extracted_pose(seq_i, img_i)
            rre, rte = get_errors(pc, patch, K, Rt, Tt, dist_thres=5)
        
    print(' > Done')


if __name__ == '__main__':
    main()