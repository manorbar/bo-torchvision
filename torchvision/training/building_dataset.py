import torch
import os
import numpy as np
import random

from PIL import Image
from tqdm import tqdm
from glob import glob
from collections import OrderedDict
from itertools import chain
from functools import reduce
from threading import Lock
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.transforms import functional as ftrans

from utils import remove_paths


class CacheLru:
    """
    threadsafe bounded size cache, removes items
    using LRU policy in order to make space when
    necessary.

    Insertion implicitly handles locking
    the cache and copying the given key and values
    into the cache.

    extraction implicitly handles locking
    and returns a copy of the data stored
    in the cache.
    """

    def __init__(self, size):
        self.cache_size = size
        self.cache = OrderedDict()
        self.cache_lock = Lock()
        return

    def __len__(self):
        return len(self.cache)

    def is_full(self):
        return len(self.cache) >= self.cache_size

    def insert(self, key, value):
        value_copy = deepcopy(value)
        self.cache_lock.acquire(blocking=True)
        if self.is_full() and key not in self.cache:
            self.cache.popitem(last=False)
        self.cache[key] = value_copy
        self.cache_lock.release()
        return

    def __getitem__(self, item):
        self.cache_lock.acquire(blocking=True)
        if item in self.cache:
            value_copy = deepcopy(self.cache[item])
            self.cache_lock.release()
            return value_copy
        else:
            self.cache_lock.release()
            return None

    def __repr__(self):
        item_string = []
        for k, v in self.cache.items():
            item_string.append(str(k) + " : " + str(v))
        return '\n'.join(item_string)


class BuildingsDataset(Dataset):
    def __init__(self, input_type: str, roots, transforms=None, validate_data=None):
        super(Dataset, self).__init__()
        self.input_type = input_type
        self.transforms = transforms
        self.roots = roots if isinstance(roots, list) else [roots]
        self.set_inputs(validate_data)

    def set_inputs(self, validate_data=None):
        """
        Creates lists of paths to the images, GT and depth directories
        """

        def assert_data_lists(paths_lst):
            len_lst = [len(plst) for plst in paths_lst if plst is not None]
            assert reduce(lambda b1, b2: b1 and b2, list(map(lambda paths_len: paths_len == len_lst[0], len_lst))), \
                f"Invalid length of inputs: {len_lst}"

        def check_input_dirs_validity():
            """
            Checks validity of dataset directories
            """

            def is_dirs(dirs_lst):
                for dirs in dirs_lst:
                    if dirs is None:
                        continue
                    for d in dirs:
                        if not os.path.isdir(d):
                            raise IsADirectoryError(
                                f"At least one of the dataset's folders not found under root directories: {dirs_lst}")

            images_dirs = [os.path.join(root_path, 'images') for root_path in self.roots if
                           os.path.exists(str(root_path))] \
                if self.input_type != "Depth" else None
            depth_dirs = [os.path.join(root_path, 'depth') for root_path in self.roots if
                          os.path.exists(str(root_path))] \
                if self.input_type != "RGB" else None
            annotations_dirs = [os.path.join(root_path, 'GT') for root_path in self.roots if
                                os.path.exists(str(root_path))]

            input_dirs = [images_dirs, annotations_dirs, depth_dirs]
            assert_data_lists(input_dirs)
            is_dirs(input_dirs)

        def validate_gt(gt_path, update_flag=False):
            gt = np.load(gt_path).squeeze(-1)
            gt_unique = np.unique(gt)
            valid_flag = True

            for object_id in gt_unique:
                if object_id == 0:  # Background
                    continue

                specific_instance = np.where(gt == object_id, 1, 0)  # Mask a specific object
                y1, x1, y2, x2 = BuildingsDataset.get_bbox(specific_instance)
                if y2 - y1 <= 0 or x2 - x1 <= 0:
                    if not update_flag:
                        valid_flag = False
                        break
                    else:  # Update GT by removing the invalid annotation
                        gt[gt == object_id] = 0
                        valid_flag = False

            if not valid_flag:
                if update_flag:
                    if len(np.unique(gt)) > 1:  # 1 for the background
                        print(f"Updating invalid sample {os.path.basename(gt_path).split('.')[0]}")
                        np.save(gt_path, gt[..., None])
                    else:  # Remove negative sample
                        print(f"Removing invalid sample {os.path.basename(gt_path).split('.')[0]}")
                        BuildingsDataset.remove_sample_by_gt_path(gt_path)
                else:
                    print(f"Found invalid sample {os.path.basename(gt_path).split('.')[0]}")

            return valid_flag

        check_input_dirs_validity()
        self.annotations_list = list()
        self.images_list, self.depth_list = list() if self.input_type != "Depth" else None, list() if self.input_type != "RGB" else None

        for i in range(len(self.roots)):
            imgs_paths = list(chain.from_iterable([glob(os.path.join(self.roots[i], 'images', f'*.{extn}')) \
                                                   for extn in ['j2k', 'jpg', 'jpeg', 'tif',
                                                                'tiff']])) if self.images_list is not None else None
            depth_paths = glob(os.path.join(self.roots[i], 'depth', f'*.npy')) if self.depth_list is not None else None
            anno_paths = glob(os.path.join(self.roots[i], 'GT', f'*.npy'))

            print(f"Add images from directory no'{i + 1}/{len(self.roots)} {self.roots[i]} to dataset")
            for j in tqdm(range(len(anno_paths))):
                validate_data_dict_flag, valid_flag = isinstance(validate_data, dict), True
                validate_data_flag = validate_data_dict_flag or isinstance(validate_data, bool)
                update_flag = validate_data.get("update_flag") if validate_data_dict_flag else False
                if validate_data_flag:
                    valid_flag = validate_gt(anno_paths[j], update_flag=validate_data.get("update_flag"))
                if valid_flag or (not valid_flag and update_flag):
                    self.annotations_list.append(anno_paths[j])
                    if self.images_list is not None: self.images_list.append(imgs_paths[j])
                    if self.depth_list is not None: self.depth_list.append(depth_paths[j])

        self.images_list = sorted(self.images_list) if self.images_list is not None else None
        self.depth_list = sorted(self.depth_list) if self.depth_list is not None else None
        self.annotations_list.sort()
        assert_data_lists([self.images_list, self.depth_list, self.depth_list])

    @staticmethod
    def remove_sample_by_gt_path(gt_path):
        splitted = gt_path.split('/')
        img_path = os.path.join('/'.join(splitted[:-2]), "images", f"{splitted[-1].split('.')[0]}.jpg")
        depth_path = os.path.join('/'.join(splitted[:-2]), "depth", f"{splitted[-1].split('.')[0]}.jpg")
        remove_paths([gt_path, img_path, depth_path])

    @staticmethod
    def get_bbox(annot_np: np.ndarray) -> (int, int, int, int):
        """
        Returns 2 coordinates that define the minimum bounding rectangle of an object
        :param annot_np: numpy.ndarray of shape (H, W) with a segmentation mask of an object
        :return: tuple of 4 int numbers in the format top, left, bottom, right
        """
        y_axis, x_axis = np.where(annot_np != 0)
        return int(np.min(y_axis)), int(np.min(x_axis)), int(np.max(y_axis)), int(
            np.max(x_axis))  # top, left, bottom, right

    @staticmethod
    def check_item_validity(target: dict) -> bool:
        """
        Function checks the validity of a transformed target and verifies that still has objects
        :param target: dict with keys used for maskrcnn training
        :return: bool
        """
        mask_sum = torch.sum(target['masks'])
        box_sum = torch.sum(target['boxes'])
        return mask_sum > 4 and box_sum > 0 and len(target['labels']) > 0

    @staticmethod
    def get_gt_target_arrays(gt: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Gets the "target" dict from a numpy array annotations object
        :param gt: ndarray of annotations
        :return: 4 ndarrays with the required data as the maskrcnn "target" dict
        """

        gt = gt.squeeze(-1)

        # Get total amount of instances using np.unique, as each of the ground truth annotations has a unique intensity value (object ID) in a single image
        obj_ids = np.unique(gt)
        boxes, labels, masks = list(), list(), list()

        for obg_id in obj_ids:
            if obg_id == 0:  # Background
                continue

            # Mask a specific index
            specific_instance = np.where(gt == obg_id, 1, 0)
            y1, x1, y2, x2 = BuildingsDataset.get_bbox(specific_instance)

            boxes.append(np.array(
                (x1, y1, x2, y2)))  # left, top, right, bottom - as the pretrained mrcnn model was trained before
            labels.append(1)  # Buildings class
            masks.append(specific_instance)

        boxes = np.stack(boxes)
        labels = np.stack(labels)
        masks = np.stack(masks)

        # Get the area of the object
        areas = masks.sum(axis=(1, 2))

        return boxes, labels, masks, areas

    def item_factory(self, tensor_image, depth_image, target):
        if self.input_type == "Combined":
            return {"main_input": tensor_image, "additional_input": depth_image, "targets": target}
        elif self.input_type == "RGBD":
            return {"main_input": torch.cat([tensor_image, depth_image], dim=0), "targets": target}
        elif self.input_type == "Depth":
            return {"main_input": depth_image, "targets": target}
        elif self.input_type == "RGB":
            return {"main_input": tensor_image, "targets": target}
        else:
            raise Exception(f"Invalid input_type={self.input_type}")

    def __getitem__(self, item: int) -> (torch.Tensor, dict):
        gt = np.load(self.annotations_list[item])
        pil_img = Image.open(self.images_list[item]).convert("RGB") if self.input_type != "Depth" else None
        depth_img = np.load(self.depth_list[item]).squeeze(-1) if self.input_type != "RGB" else None
        tensor_image = ftrans.to_tensor(pil_img).float() if self.input_type != "Depth" else None
        tensor_depth = torch.from_numpy(depth_img[None, ...]).float() if self.input_type != "RGB" else None
        boxes, labels, masks, areas = BuildingsDataset.get_gt_target_arrays(gt)

        # build_target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.int32)
        areas = torch.as_tensor(areas, dtype=torch.int32)
        target = {"boxes": boxes, "labels": labels, "masks": masks, "area": areas}

        # Transformations
        if self.transforms and boxes.size(0) > 0:
            match_hist_ref = np.array(Image.open(self.images_list[random.randint(0, len(self) - 1)]).convert("RGB"))
            image_trans, target_trans, depth_trans = self.transforms(target, pil_img, depth_img, match_hist_ref)
            tensor_image, target, tensor_depth = (image_trans, target_trans, depth_trans) \
                if (BuildingsDataset.check_item_validity(target_trans) if self.transforms.is_train else True) \
                else (tensor_image, target, tensor_depth)

        gt_path = os.path.basename(self.annotations_list[item])
        sample_name = gt_path[:gt_path.rfind('.')]

        return self.item_factory(tensor_image, tensor_depth, target), sample_name

    def __len__(self):
        return len(self.annotations_list)


def get_building_dataset(root, trans, input_type, validate_data=None):
    """
    for description of parameters, see the buildings dataset constructer docstring.
    """
    return BuildingsDataset(input_type, root, trans, validate_data)


"""
Registry maps names of pytorch datasets to functions that take kwargs and pass them to the construcer.
Use the dataset_params section of the config file (for train and test respectivly) in order to determin
the parameters passed to the constructor.

Mapped functions must return a class that inherits from the pytorch Dataset class, or implements its required 
function. The dataset must take a trans arguement, for applying transformations to the dataset. If no transform
is required, pass None.
"""
dataset_registry = {
    "buildings_dataset": get_building_dataset,
}
