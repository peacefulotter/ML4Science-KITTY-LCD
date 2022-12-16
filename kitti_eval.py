import os
import json
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict

from lcd.models import *
from lcd.losses import *

from lcd.kitti.dataset import KittiDataset

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

    dataset = KittiDataset(args["root"], mode="test")
    loader = data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
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
        x = [x.to(device) for x in batch]
        y0, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1])

        print(x[0].shape, x[1].shape)
        print(y0.shape, z0.shape)
        print(y1.shape, z1.shape)
        # R, t = rigid_transform_3D(A, B)
        
    print(' > Done')


if __name__ == '__main__':
    main()