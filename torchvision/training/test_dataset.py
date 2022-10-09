import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from torchvision.training.utils import create_image_mask_overlay_inference, collate_dict_fn, torch2numpy_image, load_yaml, mkdirs
from torchvision.training.runtime_utils import load_datasets


def visualize_data(data_loader, out_dir, mode):
    out_dir = os.path.join(out_dir, mode)
    mkdirs(out_dir)
    for data_loader_idx, data in enumerate(tqdm(data_loader)):
        images = data["main_input"]
        targets = data["targets"]
        sample_names = data["sample_name"]
        for batch_idx, (img, tar, sn) in enumerate(zip(images, targets, sample_names)):
            cv2.imwrite(
                os.path.join(out_dir, f"{sn}.jpg"),
                cv2.cvtColor(cv2.normalize(torch2numpy_image(create_image_mask_overlay_inference(img, tar["masks"])),
                                           np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                             cv2.COLOR_RGB2BGR)
            )


def main(config, dest_path):
    train_ds, valid_ds = load_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config['train_params']['batch_size'], collate_fn=collate_dict_fn,
                              shuffle=False, num_workers=4)
    valid_sampler = RandomSampler(valid_ds, replacement=True)
    valid_loader = DataLoader(valid_ds, sampler=valid_sampler, collate_fn=collate_dict_fn)
    visualize_data(train_loader, dest_path, "train")
    visualize_data(valid_loader, dest_path, "val")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MRCNN Dataset / Dataloader Tester and Visualizer")
    parser.add_argument("-c", "--cfg", type=str, required=True, help="The default MRCNN .yaml configuration file path")
    parser.add_argument("-d", "--dest", type=str, required=True, help="The destination dataset visualizations' path")
    args = vars(parser.parse_args())
    cfg = load_yaml(args["cfg"])
    main(cfg, args["dest"])
