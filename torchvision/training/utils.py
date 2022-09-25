from copy import deepcopy
import os
import re
import cv2
import yaml
import io
import matplotlib.pyplot as plt
import torch
import numpy as np
import colorsys
import random
from PIL import Image
from functools import reduce
from torchvision.transforms import ToTensor


def load_state_dict_layer_by_layer(model, pretrained_checkpoint_state_dict):
    """
    Load pretrained matched weights from a given checkpoint state_dict.
    :param model: nn.module
    :param pretrained_checkpoint_state_dict: dict, the state_dict of the pretrained model to load its parameters
    i.e.: ["backbone.fpn", "rpn", "roi_heads.box_heads", "roi_heads.mask_head", "roi_heads.mask_predictor"]
    :return: the model after loading the matching parts
    """

    num_matches = 0
    model_dict = deepcopy(model.state_dict())

    # Iterate over all layers of both models, the one to load weights to and the given checkpoint,
    # and update model's weights in each layer by the pretrained checkpoint's compatible layer,
    # if the names and sizes are fit
    chkpt_keys = pretrained_checkpoint_state_dict.keys()
    model_keys = model_dict.keys()

    for k in model_keys:
        if k in chkpt_keys:
            if model_dict[k].shape == pretrained_checkpoint_state_dict[k].shape:
                model_dict[k] = pretrained_checkpoint_state_dict[k]
                num_matches = 1

    # Load state dict with all changes
    model.load_state_dict(model_dict)
    print(f"Succeeded to load {num_matches}/{len(model_keys)} parameters from the pretrained state dict")

    return model


def load_state_dict_by_parts(model, pretrained_checkpoint_state_dict, model_parts):
    """
    Load pretrained parts from a given checkpoint state_dict.
    This function attempts to load model state dict parts according to model_parts.
    If the parameters names/weight sizes do not mach, function will print failure and continue.
    :param model: nn.module
    :param pretrained_checkpoint_state_dict: dict, the state_dict of the pretrained model to load its parameters
    :param model_parts: list of strings defining the parts to load from pretrained state dict (defined by str headers)
    i.e.: ["backbone.fpn", "rpn", "roi_heads.box_heads", "roi_heads.mask_head", "roi_heads.mask_predictor"]
    :return: the model after loading the matching parts
    """

    model_dict = deepcopy(model.state_dict())

    # Iterate over model_parts and attempt to load state dict if the names and sizes are fit
    for model_part in model_parts:
        # filter state dictionaries by model_part prefix
        filtered_pretrained_dict = {k: v for k, v in pretrained_checkpoint_state_dict.items() if
                                    k in model_dict and k.startswith(model_part)}
        filtered_model_dict = {k: v for k, v in model_dict.items() if k in model_dict and k.startswith(model_part)}

        # If the length of the pretrained params to be changed to is 0 or the amounts of params do not match,
        # skip this param
        if len(filtered_pretrained_dict.keys()) == 0 or len(filtered_pretrained_dict.keys()) != len(
                filtered_model_dict):
            print(f"Failed to load pretrained state dict for {model_part}. skipping ...")
            continue

        # Update model
        model_dict.update(filtered_pretrained_dict)

        # Load state dict with all changes
        model.load_state_dict(model_dict)
        print(f"Succeeded to load pretrained state dict for {model_part}")

    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_models(*models):
    for m in models:
        if m is not None:
            freeze_model(m)


def de_standardization(arr, mean, std):
    return arr * mean + std


def np_de_standardization(arr, mean, std):
    for ch in range(arr.shape[-1]):
        arr[..., ch] = de_standardization(arr[..., ch], mean[ch], std[ch])
    return arr


def torch_de_standardization(arr, mean, std):
    for ch in range(arr.shape[0]):
        arr[ch, ...] = de_standardization(arr[ch, ...], mean[ch], std[ch])
    return arr


def torch2numpy_image(torch_img):
    t2n = torch_img.clone().detach().cpu().numpy()
    return t2n.transpose([1, 2, 0]) if len(t2n.shape) == 3 else t2n if len(t2n.shape) == 2 else t2n.transpose(
        [0, 2, 3, 1])


def numpy2torch_image(np_img):
    return torch.from_numpy(
        np_img.transpose(2, 0, 1) if len(np_img.shape) == 3 else np_img if len(np_img.shape) == 2 else np_img.transpose(
            0, 3, 1, 2)).cpu()


def vis_masks_and_save_image(img, masks, out_path):
    cv2.imwrite(
        out_path,
        cv2.cvtColor(
            cv2.normalize(torch2numpy_image(
                create_image_mask_overlay_inference(
                    img,
                    masks,
                    alpha=0.5,
                )
            ),
                np.zeros(img.shape),
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
    )


def vis_torch_masks_and_save(img, masks, out_path):
    cv2.imwrite(
        out_path,
        cv2.cvtColor(
            cv2.normalize(
                torch2numpy_image(
                    create_image_mask_overlay_inference(
                        img,
                        masks,
                        alpha=0.5,
                    )
                ),
                np.zeros(img.shape),
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
    )


def save_torch_depth(depth, out_path):
    cv2.imwrite(
        out_path,
        cv2.cvtColor(
            cv2.normalize(
                torch2numpy_image(
                    depth
                ),
                np.zeros(depth.shape),
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
    )


def put_text(img, text):
    img = cv2.cvtColor(
        cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    cv2.putText(
        img,
        text,
        (int(img.shape[1] * 0.01), int(img.shape[0] * 0.1)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=int(img.shape[1] * 0.001),
        color=(0, 0, 255),
        thickness=int(img.shape[1] * 0.002)
    )
    return img


def get_grads(model):
    grads = list()
    for layer_name, param in model.named_parameters():
        if param.requires_grad:
            grads.append(param.grad.abs().mean())
        else:
            print(f"Layer {layer_name} does not require gradients")
            grads.append(0)
    return grads


def vis_grads(grads_lst):
    max_grad_loc = reduce(lambda max_grad1, max_grad2: max(max_grad1, max_grad2), [max(grads) for grads in grads_lst])
    for grads in grads_lst:
        plt.plot(grads)
    plt.ylabel("Mean Absolute Gradients")
    plt.xlabel("Layers")
    plt.text(int(0.1 * len(grads_lst[-1])), 0.9 * max_grad_loc, f"Maximum: {max(grads_lst[-1])}", fontsize=9)
    plt.text(int(0.1 * len(grads_lst[-1])), 0.8 * max_grad_loc, f"Minimum: {min(grads_lst[-1])}", fontsize=9)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.clf()
    return ToTensor()(Image.open(buf))


def remove_paths(paths):
    for p in paths:
        os.remove(p)


def mkdirs(*dirs):
    """
    Make directories in the given directories' paths
    :param dirs: *str, directory paths arguments
    """
    for d in dirs:
        if d is None:
            continue
        if not os.path.exists(d):
            os.makedirs(d)


def load_yaml(cfg_path):
    """
    Loads yaml file as dictionary, while fixing the yaml exponential floats parsing bug
    :param cfg_path: str, path to a yaml configuration file
    :return: dict, dictionary containing the yaml configuration
    """
    correct_loader = yaml.FullLoader
    correct_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?|
                [-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|
                \\.[0-9_]+(?:[eE][-+][0-9]+)?|
                [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|
                [-+]?\\.(?:inf|Inf|INF)|
                \\.(?:nan|NaN|NAN)
            )$''', re.X),
        list(u'-+0123456789.')
    )
    with open(cfg_path, 'r') as cfg_f:
        return yaml.load(cfg_f, Loader=correct_loader)


def create_empty_dict_key_if_required(input_dict, key):
    """
    adds a key to a dict if doesn't already exist
    :param input_dict: dict, where key should be added if required
    :param key: hashable object used as key
    """
    if key not in input_dict:
        input_dict[key] = {}


def random_colors(n_colors, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    :param n_colors: number of colors
    :param bright: boolean signaling if to use brighter color
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n_colors, 1, brightness) for i in range(n_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def second_to_time_string(seconds):
    hours = int(seconds // (60 * 60))
    minutes = int((seconds % (60 * 60)) // 60)
    seconds = int((seconds % (60 * 60)) % 60)
    elapsed_string = "{}:{}:{}".format(hours, minutes, seconds)
    return elapsed_string


def denormalize_image_tensor(tensor, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
    """
    images are nomralized using a mean dataset pixel when passed
    to the model. The image that is used is not transformed back
    when the model is done, so if we want to look at it properly
    we must "denormalize it".
    normalization for tensor t, mean m and std s: t.sub(m).div(s)
    denormalization for the same: t.mul(s).add(m)
    :param tensor: image tensor (C, H, W)
    :param image_mean: tuple or array of length 3
    :param image_std: tuple or array of length 3
    :return: tensor after inverse normalization
    """
    ten_copy = tensor.clone()
    for t, m, s in zip(ten_copy, image_mean, image_std):
        t.mul_(s).add_(m)
    return ten_copy


def create_image_mask_overlay_inference(img, masks, alpha=0.5, thresh=0.5):
    """
    takes an image as a torch tensor and the returned dict
    from running torch model on the image.
    overlays the masks onto a copy of the image with random colors
    and opacity (alpha).
    :param img: torch tesnor
    :param masks: masks are torch tensor: uint8tensor[N, 1, H, W]
    :param alpha: float in the range [0,1]
    :param thresh: float [0,1] for mask decision
    :return: overlayed image as torch tensor (C,H,W)
    """
    is_pred = False
    if len(masks.size()) == 4:
        is_pred = True
    if masks.size(0) == 0:
        return img
    # convert from normalized image to [0,255] rgb range.
    img = img.detach().cpu().clone().numpy().transpose(1, 2, 0)
    img = cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX).astype(np.int32)
    # create empty array for overlay image
    overlay = np.zeros_like(img[..., :3], dtype=np.float32)
    # add all the masks to a single overlay canvas
    colors = random_colors(masks.shape[0])
    for i in range(masks.shape[0]):
        if is_pred:
            overlay[np.where(masks[i, 0, :, :].data.numpy() > thresh)] = colors[i]
        else:
            overlay[np.where(masks[i, :, :].data.numpy() > thresh)] = colors[i]
    # create overlay
    over_img = (alpha * img[..., :3] + (1 - alpha) * overlay * 255).astype(np.int32)
    # convert to torch image: [H,W,C] -> [C,H,W] + int -> float + [0,255] -> [0, 1]
    over_img = torch.from_numpy(over_img.transpose(2, 0, 1).astype(np.float32) / 255)
    return over_img


def create_image_mask_overlay_by_sizes(img, tar, alpha=0.5, thresh=0.5, sizes_thresh=(32 ** 2, 92 ** 2)):
    """
    takes an image as a torch tensor and the returned dict
    from running torch model on the image.
    overlays the masks onto a copy of the image with random colors
    and opacity (alpha).
    :param img: torch tesnor
    :param tar: dict including masks, masks are torch tensor: uint8tensor[N, 1, H, W], and
    :param alpha: float in the range [0,1]
    :param thresh: float [0,1] for mask decision
    :param sizes_thresh: tuple of ints , size thresh
    :return: overlayed image as torch tensor (C,H,W)
    """

    is_pred = False
    if len(tar['masks'].size()) == 4:
        is_pred = True
        t = np.where(tar['masks'].detach().cpu().numpy() > 0.5, 1, 0)
        tar['area'] = t.sum(axis=(1, 2, 3))
    # convert from normalized image to [0,255] rgb range.
    img = (img.detach().cpu().clone().numpy().transpose(1, 2, 0) * 255).astype(np.int32)
    # create empty array for overlay image
    overlay = np.zeros_like(img, dtype=np.float32)
    # add all the masks to a single overlay canvas
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    for i in range(tar['masks'].shape[0]):
        if tar['area'][i] < sizes_thresh[0]:
            size = 0
        elif tar['area'][i] > sizes_thresh[1]:
            size = 2
        else:
            size = 1

        if is_pred:
            overlay[np.where(tar['masks'][i, 0, :, :].data.detach().cpu().numpy() > thresh)] = colors[size]
        else:
            overlay[np.where(tar['masks'][i, :, :].data.numpy() > thresh)] = colors[size]
    # create overlay
    over_img = (alpha * img + (1 - alpha) * overlay).astype(np.int32)
    # convert to torch image: [H,W,C] -> [C,H,W] + int -> float + [0,255] -> [0, 1]
    over_img = torch.from_numpy(over_img.transpose(2, 0, 1).astype(np.float32) / 255)
    return over_img


def create_image_mask_overlay_by_classes(img, tar, alpha=0.5, thresh=0.5):
    """
    takes an image as a torch tensor and the returned dict
    from running torch model on the image.
    overlays the masks onto a copy of the image with random colors
    and opacity (alpha).
    :param img: torch tesnor
    :param tar: dict including masks, masks are torch tensor: uint8tensor[N, 1, H, W], and
    :param alpha: float in the range [0,1]
    :param thresh: float [0,1] for mask decision
    :return: overlayed image as torch tensor (C,H,W)
    """
    is_pred = False

    if len(tar['masks'].size()) == 4:
        is_pred = True

    # convert from normalized image to [0,255] rgb range.
    img = (img.detach().cpu().clone().numpy().transpose(1, 2, 0) * 255).astype(np.int32)
    # create empty array for overlay image
    overlay = np.zeros_like(img, dtype=np.float32)
    # add all the masks to a single overlay canvas
    # TODO: add support for variant class number
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    for i in range(tar['masks'].shape[0]):
        if is_pred:
            overlay[np.where(tar['masks'][i, 0, :, :].data.cpu().numpy() > thresh)] = colors[tar['labels'][i] - 1]
        else:
            overlay[np.where(tar['masks'][i, :, :].data.cpu().numpy() > thresh)] = colors[tar['labels'][i] - 1]
    # create overlay
    over_img = (alpha * img + (1 - alpha) * overlay).astype(np.int32)
    # convert to torch image: [H,W,C] -> [C,H,W] + int -> float + [0,255] -> [0, 1]
    over_img = torch.from_numpy(over_img.transpose(2, 0, 1).astype(np.float32) / 255)
    return over_img


def collate_fn(batch):
    """
    this function is called by the dataloader in order to
    collect samples from the dataset into a single collection
    representing a batch.
    """
    return tuple(zip(*batch))


def collate_dict_fn(batch_dicts):
    """
    This function is called by the dataloader in order to
    collect samples from the dataset into a single collection
    representing a batch.
    """
    final_input_dict, sample_names = dict(), list()
    for input_dict, sample_name in batch_dicts:
        for k in input_dict.keys():
            if final_input_dict.get(k) is None:
                final_input_dict[k] = list()
            final_input_dict[k].append(input_dict[k])
        sample_names.append(sample_name)
    for k in final_input_dict.keys():
        final_input_dict[k] = tuple(final_input_dict[k])
    sample_names = tuple(sample_names)
    return final_input_dict, sample_names
