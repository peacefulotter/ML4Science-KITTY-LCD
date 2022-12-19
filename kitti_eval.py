import os
import json
import torch
import torch.utils.data as data

from lcd.models import *
from lcd.losses import *

from lcd.kitti.eval_dataset import KittiEvalDataset
from lcd.descriptors import find_descriptors_correspondence

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
        num_workers=1, # args["num_workers"],
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
        print(f'[{i}] Forwading to model')
        print(batch[0].shape, batch[1].shape)
        x = [torch.flatten(x, end_dim=1).to(device) for x in batch]
        print(x[0].shape, x[1].shape)
        _, z0 = pointnet(x[0])
        _, z1 = patchnet(x[1])

        print(f'[{i}] Establishing descriptor correspondences')
        correspondences = find_descriptors_correspondence(z0[:100], z1[:100])
        print(correspondences.shape)
        print(correspondences)
        # R, t = rigid_transform_3D(A, B)
        
    print(' > Done')


if __name__ == '__main__':
    main()