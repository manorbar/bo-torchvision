"""
Set of torch compatible transfromation classes for training and testing datasets.

"""
import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import functional as ftrans
from copy import deepcopy
from PIL import Image
from functools import reduce


class BuildingTransform:
    def __init__(self, config, train=True):
        """
        A class that should be used to transform images and targets when using
        the BuildingDataset for training. Implements the transformations
        on the image and targets using torch.transforms.functional module.

        implemented transforms for images and targets:
            - flip left right
            - flip up down
            - rotation
            - scale
            - translate
            - shear

        as most transformations and augementations on images must be applied
        to the masks as well, i chose to randomize the augmentations  parameters myself
        and apply them to the images and tensors "manually" using the functional transforms.

        As most of the torch transformations are intended for use on PIL images, i had
        to convert the mask tensors to pil images, augment / transform, and then convert
        back to tensors. This might waste time and some space, but i did it because:
            - that way i know the same torch transform is being carried out
              on the image and the tensor
            - i assume the torch transformations are faster than the imaug library or
              something i might write myself.
            - keep torch as the only major dependency for this code, no need for opencv
              or additional open source projects (like cocotools, imaug etc).

        :param config: dictionary of transformation and augmentation parameters.
        """
        self.params = config
        self.is_train = train
        return

    @staticmethod
    def hflip_masks(pil_masks):
        for i in range(len(pil_masks)):
            pil_masks[i] = ftrans.hflip(pil_masks[i])
        return pil_masks

    @staticmethod
    def vflip_masks(pil_masks):
        for i in range(len(pil_masks)):
            pil_masks[i] = ftrans.vflip(pil_masks[i])
        return pil_masks

    @staticmethod
    def rotate_masks(pil_masks, angle):
        for i in range(len(pil_masks)):
            pil_masks[i] = ftrans.rotate(pil_masks[i], angle)
        return pil_masks

    @staticmethod
    def resized_crop_masks(pil_masks, i, j, h, w, size, interpolation):
        for i in range(len(pil_masks)):
            if isinstance(pil_masks[i], Image.Image):
                pil_masks[i] = ftrans.resized_crop(pil_masks[i], i, j, h, w, size, interpolation)
        return pil_masks

    @staticmethod
    def translate_masks(pil_masks, trans_x, trans_y):
        for i in range(len(pil_masks)):
            pil_masks[i] = ftrans.affine(pil_masks[i], 0, (trans_x, trans_y), 1, 0)
        return pil_masks

    @staticmethod
    def shear_masks(pil_masks, angle):
        for i in range(len(pil_masks)):
            pil_masks[i] = ftrans.affine(pil_masks[i], 0, (0, 0), 1, angle)
        return pil_masks

    @staticmethod
    def convert_pil_masks_to_tensor(pil_masks):
        width, height = pil_masks[0].size
        non_empty_tensor_list = []
        non_empty_indices = []
        for i in range(len(pil_masks)):
            temp_tens = ftrans.to_tensor(pil_masks[i])[0, :, :]
            if torch.sum(temp_tens) >= 5:
                non_empty_tensor_list.append(temp_tens)
                non_empty_indices.append(i)

        # populate complete mask tensor
        if len(non_empty_tensor_list) > 0:
            tensor_masks = torch.zeros((len(non_empty_tensor_list), height, width))
            for i in range(len(non_empty_tensor_list)):
                tensor_masks[i, :, :] = non_empty_tensor_list[i]
            return tensor_masks, non_empty_indices
        else:
            return torch.zeros((0, 1, 1)), non_empty_indices  # TODO this is an ugly hack

    @staticmethod
    def find_mask_bounding_boxes(tensor_masks):
        d, r, c = np.where(tensor_masks)
        boxes = []
        boxes_index = []
        for i in np.unique(d):
            min_r = np.min(r[np.where(d == i)])
            max_r = np.max(r[np.where(d == i)])
            min_c = np.min(c[np.where(d == i)])
            max_c = np.max(c[np.where(d == i)])
            if (np.abs(max_r - min_r) > 0) and (np.abs(max_c - min_c) > 0):
                boxes.append(torch.Tensor((min_c, min_r, max_c, max_r)))
                boxes_index.append(i)

        if boxes:
            return torch.stack(boxes), boxes_index
        else:
            return torch.zeros((0, 4)), boxes_index  # TODO sort this out, it works but its ugly

    @staticmethod
    def get_hist(img):
        hist = list()
        for c in range(img.shape[2]):
            hist.append(np.histogram(img[..., c], bins=256, range=(0, 255))[0])
        return np.stack(hist, axis=0)

    @staticmethod
    def match_histogram(src, ref, ref_weight=1, thresh=5e-4):
        ref_hist = BuildingTransform.get_hist(ref)
        src_hist = BuildingTransform.get_hist(src)

        if reduce(lambda b1, b2: b1 and b2,
                  [ref_hist_c[0] < (ref.shape[0] * ref.shape[1]) * thresh for ref_hist_c in ref_hist]) and reduce(
            lambda b1, b2: b1 and b2,
            [src_hist_c[0] < (src.shape[0] * src.shape[1]) * thresh for src_hist_c in src_hist]):
            matched = np.empty(src.shape, dtype=np.uint8)

            for c in range(src.shape[-1]):
                _, src_unique_indices, src_counts = np.unique(src[..., c].ravel(), return_inverse=True,
                                                              return_counts=True)
                src_quantiles = np.cumsum(src_counts).astype(np.float64) / src.size
                tmpl_quantiles = np.cumsum(ref_hist[c]).astype(np.float64) / ref.size
                interp_a_values = np.interp(src_quantiles, tmpl_quantiles, [i for i in range(len(ref_hist[c]))])
                _, src_unique_indices = np.unique(src[..., c].ravel(), return_inverse=True)
                matched[..., c] = ref_weight * interp_a_values[src_unique_indices].reshape(src.shape[:-1]) + (
                        1 - ref_weight) * src[..., c]

            return matched

        return src

    @staticmethod
    def random_match_hist(img, ref, ref_weight_range=(0.4, 0.65), zero_padding_thresh=5e-4):
        return BuildingTransform.match_histogram(img, ref, random.uniform(ref_weight_range[0], ref_weight_range[1]),
                                                 zero_padding_thresh)

    def train_call(self, target, img=None, depth=None, match_hist_ref=None):
        # list of transformations to use, we use this for randomizing the order of augmentation
        ops_applied = False
        ops = [op for op in self.params.keys() if op not in ["normalize", "clahe", "hist_matching"] if
               self.params[op]['use']]
        random.shuffle(ops)

        # create list of pil masks
        num_masks = target['masks'].shape[0]
        pil_masks = [ftrans.to_pil_image(target['masks'][i, :, :].numpy()[:, :, np.newaxis]) for i in range(num_masks)]

        # Depth normalization
        if depth is not None and self.params["normalize"]['depth']['use']:
            depth = ((depth - depth.min()) / (depth.max() - depth.min()))
            # depth = ((depth - depth.min()) / self.params["normalize"]['depth_normalize_factor'])

        pil_depth = ftrans.to_pil_image((depth * 255).astype(np.uint8)) if depth is not None else None

        # Randomly apply histogram matching at the beginning. If applied - do not apply more color augmentations afterwards to preserve the histogram matching effect
        hist_matching_flag = random.random() <= self.params['hist_matching'][
            'probability'] and img is not None and match_hist_ref is not None
        if hist_matching_flag:
            img = Image.fromarray(BuildingTransform.random_match_hist(np.array(img),
                                                                      match_hist_ref, tuple(
                    self.params['hist_matching']["ref_weight_range"]), self.params['hist_matching'][
                                                                          "zero_padding_thresh"]))

        # transformations in random order
        for op in ops:
            # random horizontal flip
            if op == 'hflip' and random.random() <= self.params['hflip']["probability"]:
                img = ftrans.hflip(img) if img is not None else None
                pil_depth = ftrans.hflip(pil_depth) if pil_depth is not None else None
                pil_masks = BuildingTransform.hflip_masks(pil_masks)
                ops.pop(ops.index('hflip'))
                ops_applied = True

            # random vertical flip
            if op == 'vflip' and random.random() <= self.params['vflip']["probability"]:
                img = ftrans.vflip(img) if img is not None else None
                pil_depth = ftrans.vflip(pil_depth) if pil_depth is not None else None
                pil_masks = BuildingTransform.vflip_masks(pil_masks)
                ops.pop(ops.index('vflip'))
                ops_applied = True

            # random rotate
            if op == "rotate" and random.random() <= self.params['rotate']["probability"]:
                rand_angle = np.random.uniform(self.params['rotate']['min_angle'], self.params['rotate']['max_angle'])
                img = ftrans.rotate(img, rand_angle) if img is not None else None
                pil_depth = ftrans.rotate(pil_depth, rand_angle) if pil_depth is not None else None
                pil_masks = BuildingTransform.rotate_masks(pil_masks, rand_angle)
                ops_applied = True

            # random translation
            if op == "translate" and random.random() <= self.params['translate']["probability"]:
                width, height = img.size if img is not None else pil_depth.size
                rand_x = np.random.uniform(self.params['translate']['min_x_pct'], self.params['translate']['max_x_pct'])
                rand_y = np.random.uniform(self.params['translate']['min_y_pct'], self.params['translate']['max_y_pct'])
                rand_x_trans = int(rand_x * width)
                rand_y_trans = int(rand_y * height)
                img = ftrans.affine(img, 0, (rand_x_trans, rand_y_trans), 1, 0) if img is not None else None
                pil_depth = ftrans.affine(pil_depth, 0, (rand_x_trans, rand_y_trans), 1,
                                          0) if pil_depth is not None else None
                pil_masks = BuildingTransform.translate_masks(pil_masks, rand_x_trans, rand_y_trans)
                ops_applied = True

            # random shear
            if op == "shear" and random.random() <= self.params['shear']["probability"]:
                rand_angle = np.random.uniform(self.params['shear']['min_angle'], self.params['shear']['max_angle'])
                img = ftrans.affine(img, 0, (0, 0), 1, rand_angle) if img is not None else None
                pil_depth = ftrans.affine(pil_depth, 0, (0, 0), 1, rand_angle) if pil_depth is not None else None
                pil_masks = BuildingTransform.shear_masks(pil_masks, rand_angle)
                ops_applied = True

            # random blur
            if op == "blur" and random.random() <= self.params['blur']["probability"]:
                k = tuple(random.sample(self.params['blur']["kernels"], 1)[0])
                img = Image.fromarray(cv2.GaussianBlur(np.array(img), k, 0)) if img is not None else None
                pil_depth = Image.fromarray(
                    cv2.GaussianBlur(np.array(pil_depth), k, 0)) if pil_depth is not None else None
                ops_applied = True

            # random zoom
            if op == "zoom" and random.random() <= self.params['zoom']["probability"]:
                resized_cropper = T.RandomResizedCrop(img.size, scale=tuple(self.params['zoom']["scale"]))
                i, j, h, w = resized_cropper.get_params(img, resized_cropper.scale, resized_cropper.ratio)
                img = ftrans.resized_crop(img, i, j, h, w, resized_cropper.size,
                                          resized_cropper.interpolation) if img is not None else None
                pil_depth = ftrans.resized_crop(pil_depth, i, j, h, w, resized_cropper.size,
                                                resized_cropper.interpolation) if pil_depth is not None else None
                pil_masks = BuildingTransform.resized_crop_masks(pil_masks, i, j, h, w, resized_cropper.size,
                                                                 resized_cropper.interpolation)
                ops_applied = True

            if not hist_matching_flag:
                # color transforms
                if op == "brightness" and random.random() <= self.params['brightness'][
                    'probability'] and img is not None:
                    brightness_factor = np.random.uniform(self.params['brightness']['min_level'],
                                                          self.params['brightness']['max_level'])
                    img = ftrans.adjust_brightness(img, brightness_factor)

                if op == "contrast" and random.random() <= self.params['contrast']['probability'] and img is not None:
                    contrast_factor = np.random.uniform(self.params['contrast']['min_level'],
                                                        self.params['contrast']['max_level'])
                    img = ftrans.adjust_contrast(img, contrast_factor)

                if op == "gamma" and random.random() <= self.params['gamma']['probability'] and img is not None:
                    gamma = np.random.uniform(self.params['gamma']['min_level'], self.params['gamma']['max_level'])
                    img = ftrans.adjust_gamma(img, gamma)

                if op == "hue" and random.random() <= self.params['hue']['probability'] and img is not None:
                    hue_factor = np.random.uniform(self.params['hue']['min_level'], self.params['hue']['max_level'])
                    img = ftrans.adjust_hue(img, hue_factor)

                if op == "saturation" and random.random() <= self.params['saturation'][
                    'probability'] and img is not None:
                    saturation_factor = np.random.uniform(self.params['saturation']['min_level'],
                                                          self.params['saturation']['max_level'])
                    img = ftrans.adjust_saturation(img, saturation_factor)

        # clahe
        if random.random() <= self.params['clahe']['probability'] and img is not None and not hist_matching_flag:
            img = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=self.params['clahe']['clipLimit'],
                                    tileGridSize=tuple(self.params['clahe']['tileGridSize']))
            for i in range(img.shape[-1]):
                img[..., i] = clahe.apply(img[..., i])

        # convert image back to tenser
        img = ftrans.to_tensor(img) if img is not None else None
        depth = torch.from_numpy(np.asarray(pil_depth)[None, ...] / 255).float() if pil_depth is not None else None

        # normalize
        if self.params["normalize"]['rgb']['use'] and img is not None:
            img = ftrans.normalize(img, self.params["normalize"]['rgb']['image_mean'],
                                   self.params["normalize"]['rgb']['image_std'])

        target = deepcopy(target)
        # convert masks back if operations were applied, otherwise, return the original masks
        if ops_applied:
            tensor_masks, non_empty_indices = BuildingTransform.convert_pil_masks_to_tensor(pil_masks)
            # replace old bounding boxes with transformed ones
            target['boxes'], boxes_idx = BuildingTransform.find_mask_bounding_boxes(tensor_masks)

            # replace pre transformation masks with new ones
            target['masks'] = tensor_masks[boxes_idx]

            if target['masks'].shape[0] != target['boxes'].shape[0]:
                mask_num = target['masks'].shape[0]
                print(f'mask number is {mask_num}')
                print(tensor_masks)
                boxes_num = target['boxes'].shape[0]
                print(f'boxes number is {boxes_num}')
                print(target['boxes'])

            # filter relevant labels and areas
            target['labels'] = target['labels'][non_empty_indices][boxes_idx]
            target['area'] = target['area'][non_empty_indices][boxes_idx]

        return img, target, depth

    def test_call(self, img=None, depth=None):
        # clahe
        if 'clahe' in self.params and self.params['clahe']['use'] and img is not None:
            img = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=self.params['clahe']['clipLimit'],
                                    tileGridSize=tuple(self.params['clahe']['tileGridSize']))
            for i in range(img.shape[-1]):
                img[..., i] = clahe.apply(img[..., i])

        img = ftrans.to_tensor(img) if img is not None else None

        # normalize
        if self.params["normalize"]['rgb']['use'] and img is not None:
            img = ftrans.normalize(img, self.params["normalize"]['rgb']['image_mean'],
                                   self.params["normalize"]['rgb']['image_std'])

        if depth is not None and self.params["normalize"]['depth']['use']:
            # depth = ((depth - depth.min()) / self.params["normalize"]['depth_normalize_factor'])
            dmin = depth.min()
            depth = (depth - dmin) / (depth.max() - dmin)

        depth = torch.from_numpy(depth[None, ...]).float() if depth is not None else None

        return img, depth

    def __call__(self, target, img=None, depth=None, match_hist_ref=None):
        """
        if object is set to train, call the training augmentation and transformation method.
        otherwise, call test method which only convert image from pil to tensor and norms.
        :param img: pil image
        :param target: coco compatible dictionary
        :return: torch tensor and coco compatible target dictionary
        """
        assert img is not None or depth is not None, "At least one of the input tensors should not be None"
        if self.is_train:
            return self.train_call(target, img, depth, match_hist_ref)
        img, depth = self.test_call(img, depth)
        return img, target, depth


if __name__ == '__main__':
    import os
    import json

    if os.path.isfile("config.json"):
        cfg = json.loads(open('config.json', 'r').read())
        train_trans = BuildingTransform(cfg, train=True)
        test_trans = BuildingTransform(cfg, train=False)
        print("init test passed.")
    else:
        print("no config.json found in local dir")
