from typing import Tuple, Dict

import numpy as np
import torch
from PIL import Image

from torchvision.transforms import functional as ftrans
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def get_maskrcnn_resnet_model(config):
    # parse required data from config
    num_classes, general_model_params, backbone_params, anchor_generator_params, mask_roi_pool_params = \
        config['num_classes'], config['general_model_params'], config['backbone_params'], \
        config['anchor_generator_params'], config['mask_roi_pool_params']

    '''
        Anchor generator is given as an instance to MaskRCNN constructor. When required to replace it we need to build 
        our own instanceand send it to the constructor. 
        If kept unchanged, send None to the constructor making it use the default params
    '''
    anchor_sizes = anchor_generator_params['anchor_sizes']
    aspect_ratios = anchor_generator_params['aspect_ratios']
    if anchor_sizes or aspect_ratios:
        # Generator expects to receive tuple of tuples
        # Set params if given else set to defaults (according to fasterRCNN defaults)
        anchor_sizes = tuple((x,) for x in anchor_sizes) if anchor_sizes else ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = (tuple(aspect_ratios) if aspect_ratios else (0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    else:
        anchor_generator = None

    # MultiScale RoI pooler is given as an instance to MaskRCNN constructor.
    # When required to replace it we need to build our own instance
    # and send it to the constructor. If kept unchanged, send None to the constructor making it use the default params
    featmap_names = mask_roi_pool_params['featmap_names']
    output_size = mask_roi_pool_params['output_size']
    sampling_ratio = mask_roi_pool_params['sampling_ratio']
    if featmap_names or output_size or sampling_ratio:
        # Set params if given else set to defaults (according to MaskRCNN defaults)
        featmap_names = list(map(str, featmap_names)) if featmap_names else ['0', '1', '2', '3']
        output_size = output_size if output_size else 14
        sampling_ratio = sampling_ratio if sampling_ratio else 2
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size,
                                           sampling_ratio=sampling_ratio)
    else:
        mask_roi_pool = None

    # Create a backbone instance to send to MaskRCNN constructor
    two_sided_fpn = backbone_params['two_sides_fpn'] if backbone_params.get('two_sides_fpn') else False
    backbone = resnet_fpn_backbone(backbone_params['backbone_name'], backbone_params['pretrained'],
                                   two_sides=two_sided_fpn)

    # In any case we remove this key as it's not part of the MaskRCNN constructor
    general_model_params.pop('pretrained')

    # Build model with required backbone and all other given params
    model = MaskRCNN(backbone, num_classes, **general_model_params, rpn_anchor_generator=anchor_generator,
                     mask_roi_pool=mask_roi_pool)

    # default box predictor contains 21 classes, replace with the required number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # default mask predictor has 21 options, replace with right number.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


class PyTorchNetworkAdapter:
    def __init__(self, model_args: Dict, weights_path: str):
        super().__init__()
        self.device: torch.device = torch.device('cuda')
        self.model: MaskRCNN = get_maskrcnn_resnet_model(model_args).to(self.device)
        with open(weights_path, "rb") as weights_file:
            weights: dict = torch.load(weights_file, )
            self.model.load_state_dict(weights['model_state_dict'])
            del weights
        self.model.eval()
        self.model.roi_heads.score_thresh = model_args['general_model_params']['box_score_thresh']

    @torch.no_grad()
    def forward_image_in_net(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray or None]:
        """
        forwards the image to the net
        :param image: the image to forward
        :return: a mask
        """
        tensor_image: torch.Tensor = ftrans.to_tensor(image).unsqueeze(0).to(self.device).float()
        # for some odd reason the model returns an array with 1 item which is the dictionary of the predication
        pred: Dict = self.model(tensor_image)[0]
        masks: torch.Tensor = pred['masks'].clone()
        masks_amount = pred['masks'].shape[0]
        if masks_amount == 0:
            return np.array([]), None

        # assuming 1 class every pixel with prob. above 0.5 is considered a building
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        np_masks: np.ndarray = torch.Tensor.cpu(masks).detach().numpy().astype(int)
        confs: np.ndarray = torch.Tensor.cpu(pred['scores']).detach().numpy()
        np_masks_reshaped: np.ndarray = np_masks.reshape((masks_amount, image.shape[0], image.shape[1]))
        del tensor_image, masks
        torch.cuda.empty_cache()
        return np_masks_reshaped, confs

    def remove_model_from_memory(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def load(image_path: str) -> np.ndarray:
    """
    Loads an image tile and returns a 3D array with all the channels
    normalized to between 0 and 1.
    :param image_path: absolute path to image.
    :return: np array representing the image
    """
    # convert to RGB and normalize pixels value to 0...1
    return load_without_normalize(image_path) / 255


def load_without_normalize(image_path: str) -> np.ndarray:
    """
    Loads an image tile and returns a 3D array with all the channels
    normalized to between 0 and 1.
    :param image_path: absolute path to image.
    :return: np array representing the image
    """
    # if not image_path.exists():
    #     raise Exception(f'Image not found at path {image_path}')

    return np.array(Image.open(image_path).convert('RGB'))


import json

with open('/home/weights/robust_v1.json') as f:
    model_args = json.loads(f.read())

model = PyTorchNetworkAdapter(model_args, '/home/weights/maskrcnn_resnet_base')
image: np.ndarray = load('/home/images/buildings.jpg')
preds, congs = model.forward_image_in_net(image)
print(image)

# # import necessary libraries
# from PIL import Image
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
#
# import torch
# import torchvision.transforms as T
# import torchvision
# import numpy as np
#
# import cv2
# import random
# import warnings
#
# warnings.filterwarnings('ignore')
#
# from torchvision.models.detection import maskrcnn_resnet50_fpn
#
#
# os.environ['DEVICE'] = 'cuda'
#
#
# # PyTorchNetworkAdapter({
# #
# # })
#
# def load(image_path: str) -> np.ndarray:
#     """
#     Loads an image tile and returns a 3D array with all the channels
#     normalized to between 0 and 1.
#     :param image_path: absolute path to image.
#     :return: np array representing the image
#     """
#     # convert to RGB and normalize pixels value to 0...1
#     return load_without_normalize(image_path) / 255
#
#
# def load_without_normalize(image_path: str) -> np.ndarray:
#     """
#     Loads an image tile and returns a 3D array with all the channels
#     normalized to between 0 and 1.
#     :param image_path: absolute path to image.
#     :return: np array representing the image
#     """
#     # if not image_path.exists():
#     #     raise Exception(f'Image not found at path {image_path}')
#
#     return np.array(Image.open(image_path).convert('RGB'))
#
#
# # # load model
# # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda()
# # # set to evaluation mode
# # model.eval()
# # # matplotlib.use('PyQt6')
# # # matplotlib.use('Qt5Agg')
# #
# # # load COCO category names
# # COCO_CLASS_NAMES = [
# #     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
# #     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
# #     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
# #     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
# #     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
# #     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
# #     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
# #     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
# #     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
# #     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
# #     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
# #     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# # ]
# #
# # #
# import random
#
#
# def get_coloured_mask(mask):
#     """
#     random_colour_masks
#       parameters:
#         - image - predicted masks
#       method:
#         - the masks of each predicted object is given random colour for visualization
#     """
#     colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
#                [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
#     r = np.zeros_like(mask).astype(np.uint8)
#     g = np.zeros_like(mask).astype(np.uint8)
#     b = np.zeros_like(mask).astype(np.uint8)
#     r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
#     coloured_mask = np.stack([r, g, b], axis=2)
#     return coloured_mask
# #
# #
# # def get_prediction(img_path, confidence):
# #     """
# #     get_prediction
# #       parameters:
# #         - img_path - path of the input image
# #         - confidence - threshold to keep the prediction or not
# #       method:
# #         - Image is obtained from the image path
# #         - the image is converted to image tensor using PyTorch's Transforms
# #         - image is passed through the model to get the predictions
# #         - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
# #           ie: eg. segment of cat is made 1 and rest of the image is made 0
# #
# #     """
# #     img = Image.open(img_path)
# #     transform = T.Compose([T.ToTensor()])
# #     img = transform(img).cuda()
# #     pred = model([img])
# #     pred_score = list(Tensor.cpu(pred[0]['scores']).detach().numpy())
# #     pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
# #     masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
# #     pred_class = [COCO_CLASS_NAMES[i] for i in list(Tensor.cpu(pred[0]['labels']).numpy())]
# #     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(Tensor.cpu(pred[0]['boxes']).detach().numpy())]
# #     masks = masks[:pred_t + 1]
# #     pred_boxes = pred_boxes[:pred_t + 1]
# #     pred_class = pred_class[:pred_t + 1]
# #     return masks, pred_boxes, pred_class
# #
# #
# # def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
# #     """
# #     segment_instance
# #       parameters:
# #         - img_path - path to input image
# #         - confidence- confidence to keep the prediction or not
# #         - rect_th - rect thickness
# #         - text_size
# #         - text_th - text thickness
# #       method:
# #         - prediction is obtained by get_prediction
# #         - each mask is given random color
# #         - each mask is added to the image in the ration 1:0.8 with opencv
# #         - final output is displayed
# #     """
# #     masks, boxes, pred_cls = get_prediction(img_path, confidence)
# #     img = cv2.imread(img_path)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     for i in range(len(masks)):
# #         rgb_mask = get_coloured_mask(masks[i])
# #         img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
# #         x1y1 = tuple(int(x) for x in boxes[i][0])
# #         x2y2 = tuple(int(x) for x in boxes[i][1])
# #         cv2.rectangle(img, x1y1, x2y2, color=(0, 255, 0), thickness=rect_th)
# #         cv2.putText(img, pred_cls[i], x1y1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
# #     plt.figure(figsize=(20, 30))
# #     plt.imshow(img)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.show()
# #
# #
# # segment_instance('./traffic.jpg', confidence=0.7)
