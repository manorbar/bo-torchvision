import os
import torch

from copy import deepcopy

from torchvision import models
from torchvision.training.utils import load_state_dict_layer_by_layer, freeze_models
from torchvision.training.fpn_factory import fpn_factory
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from torch.hub import load_state_dict_from_url


class MaskRCNNInitNamespace:

    @staticmethod
    def set_mrcnn_num_classes(model, num_classes):
        # Replace box predictor with required number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        # Replace mask predictor with the right number of classes
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256,
                                                                                      num_classes)

    @staticmethod
    def match_pretrained_rgb_mrcnn_state_dict_to_rgbd_late_fusion_mrcnn(pretrained_state_dict, input_type="RGB"):
        layer_types_to_update = ["rgb", "depth"] if input_type == "Combined" else ["rgb"] if input_type == "RGB" else [
            "depth"]

        def update_inter_layer(k, d, layer_types):
            for lt in layer_types:
                splitted = k.split('.')
                new_k = '.'.join(splitted[:2] + [f'inter_{lt}'] + splitted[2:])
                d[new_k] = v
            del d[k]

        new_dict = deepcopy(pretrained_state_dict)
        for k, v in pretrained_state_dict.items():
            if k.startswith(
                    'backbone.body') and 'inter_rgb' not in k and 'inter_depth' not in k and input_type == "Combined":
                update_inter_layer(k, new_dict, layer_types_to_update)
        pretrained_state_dict = new_dict
        return pretrained_state_dict

    @staticmethod
    def get_rpn_anchor_generator(anchor_generator_params):
        cfg = deepcopy(anchor_generator_params)
        if cfg.get("freeze") is not None:
            cfg.pop("freeze")
        # Anchor generator is given as an instance to MaskRCNN constructor. When required to replace it we need to build our own instance
        # and send it to the constructor. If kept unchanged, send None to the constructor making it use the default params
        anchor_generator = None
        anchor_sizes = cfg['anchor_sizes']
        aspect_ratios = cfg['aspect_ratios']
        if anchor_sizes or aspect_ratios:
            # Generator expects to receive tuple of tuples
            # Set params if given else set to defaults (according to fasterRCNN defaults)
            anchor_sizes = tuple((x,) for x in anchor_sizes) if anchor_sizes else ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = (tuple(aspect_ratios) if aspect_ratios else (0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        return anchor_generator

    @staticmethod
    def get_mask_roi_pool(mask_roi_pool_params):
        mask_roi_pool = None
        output_size = mask_roi_pool_params['output_size']
        featmap_names = mask_roi_pool_params['featmap_names']
        sampling_ratio = mask_roi_pool_params['sampling_ratio']
        if featmap_names or output_size or sampling_ratio:
            # Set params if given else set to defaults (according to MaskRCNN defaults)
            featmap_names = featmap_names if featmap_names else [0, 1, 2, 3]
            output_size = output_size if output_size else 14
            sampling_ratio = sampling_ratio if sampling_ratio else 2
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size,
                                               sampling_ratio=sampling_ratio)
        return mask_roi_pool

    @staticmethod
    def adjust_backbone_params(backbone_params, use_pretrained_mrcnn):
        assert isinstance(backbone_params.get("main_backbone"), dict), \
            f"Invalid main backbone parameters: {backbone_params.get('main_backbone')}"

        new_backbone_params = {
            "num_of_layers_in_pyramid": backbone_params.get("num_of_layers_in_pyramid"),
            "two_sides_fpn": False if backbone_params.get("two_sides_fpn") is None else backbone_params["two_sides_fpn"]
        }

        main_backbone_params = backbone_params.get("main_backbone")
        additional_backbone_params = backbone_params.get("additional_backbone")
        valid_main_backbone_params = isinstance(main_backbone_params, dict)
        valid_additional_backbone_params = isinstance(additional_backbone_params, dict)
        rgb_imagenet_pretrained = not use_pretrained_mrcnn and main_backbone_params.get(
            "pretrained") if valid_main_backbone_params else False
        depth_imagenet_pretrained = not use_pretrained_mrcnn and additional_backbone_params.get(
            "pretrained") if valid_additional_backbone_params else False

        if valid_additional_backbone_params and backbone_params.get("fusion_type"):
            new_backbone_params["rgb_backbone_params"] = main_backbone_params
            new_backbone_params["rgb_backbone_params"]["pretrained"] = rgb_imagenet_pretrained
            new_backbone_params["depth_backbone_params"] = additional_backbone_params
            new_backbone_params["depth_backbone_params"]["pretrained"] = depth_imagenet_pretrained
            return new_backbone_params

        new_backbone_params["backbone_name"] = main_backbone_params.get("name")
        new_backbone_params["pretrained"] = rgb_imagenet_pretrained
        return new_backbone_params

    @staticmethod
    def get_maskrcnn_resnet_model(config, input_type):
        """
        Get MaskRCNN model according to the given configuration dictionary, including:
        1.  Loading the configurable model type from: 'RGB', 'Depth', 'RGBD' or 'Combined' (using the configurable late-fusion type) MaskRCNN
        2.  Loading various pretrained weights from a checkpoint path or a COCO / ImageNet pretrained MaskRCNN / ResNet backbone.
            Supported loading types:
            2.1.    Loading 'RGB' or 'RGBD' MaskRCNNs to 'Combined' (late-fusion) MaskRCNN, while loading the backbone to the main backbone intermediate RGB pathway
            2.2.    Loading 'Combined' MaskRCNN to another 'Combined' one
            2.3.    Loading from each one in the set of 'RGB', 'Depth' and 'RGBD' MaskRCNNs to another
            In order to load a Depth backbone from a Depth MaskRCNN to a 'Combined' MaskRCNN, we need to save the backbone's
            stand-alone weights and set the relevant configuration backbone_params-->additional_backbone-->pretrained in the configuration file with their filepath
        :param config: dict, a configuration dictionary
        :param input_type: str, one of the following strings: 'RGB', 'Depth', 'RGBD' or 'Combined'
        :return: torch.nn.Module, the returned configurable model
        """

        cfg = deepcopy(config)

        # Anchor generator is given as an instance to MaskRCNN constructor. When required to replace it we need to build our own instance
        # and send it to the constructor. If kept unchanged, send None to the constructor making it use the default params
        anchor_generator = MaskRCNNInitNamespace.get_rpn_anchor_generator(cfg['anchor_generator_params'])

        # MultiScale RoI pooler is given as an instance to MaskRCNN constructor. When required to replace it we need to build our own instance
        # and send it to the constructor. If kept unchanged, send None to the constructor making it use the default params
        mask_roi_pool = MaskRCNNInitNamespace.get_mask_roi_pool(cfg['mask_roi_pool_params'])

        pretrained_mrcnn = None
        assert isinstance(cfg.get("general_model_params"), dict), "Invalid configuration file"
        try:  # Dealing with the edge case when cfg["general_model_params"]["pretrained_mrcnn"] exists but is unset
            pretrained_mrcnn = cfg["general_model_params"]["pretrained_mrcnn"]
            cfg["general_model_params"].pop("pretrained_mrcnn")
        except:
            pass

        use_pretrained_mrcnn = isinstance(pretrained_mrcnn, str) and os.path.exists(str(pretrained_mrcnn))

        print(f"Creating MaskRCNN model")
        model = MaskRCNN(
            fpn_factory(
                backbone_params=MaskRCNNInitNamespace.adjust_backbone_params(cfg['backbone_params'],
                                                                             use_pretrained_mrcnn),
                input_type=input_type,
                fusion_type=config["backbone_params"].get("fusion_type") if config.get("backbone_params") else None
            ),
            cfg['num_classes'],
            **cfg['general_model_params'],
            rpn_anchor_generator=anchor_generator,
            mask_roi_pool=mask_roi_pool
        )

        if config['train_params'].get('resume'):
            return model

        # Loading required pretrained MaskRCNN model
        if use_pretrained_mrcnn:
            print(f"Loading pretrained state dict from {pretrained_mrcnn}")
            model = load_state_dict_layer_by_layer(
                model=model,
                pretrained_checkpoint_state_dict=MaskRCNNInitNamespace.match_pretrained_rgb_mrcnn_state_dict_to_rgbd_late_fusion_mrcnn(
                    pretrained_state_dict=torch.load(pretrained_mrcnn)['model_state_dict'],
                    input_type=input_type
                )
            )
        elif pretrained_mrcnn:
            mrcnn_pretrained_url = model_urls['maskrcnn_resnet50_fpn_coco']
            print(f"Loading pretrained state dict from url {mrcnn_pretrained_url}")
            model = load_state_dict_layer_by_layer(
                model=model,
                pretrained_checkpoint_state_dict=MaskRCNNInitNamespace.match_pretrained_rgb_mrcnn_state_dict_to_rgbd_late_fusion_mrcnn(
                    pretrained_state_dict=load_state_dict_from_url(mrcnn_pretrained_url, progress=True),
                    input_type=input_type
                )
            )

        for param in model.parameters():
            param.requires_grad = True

        # Freeze required model layers
        freeze_models(
            model.backbone.body.inter_rgb if input_type == "Combined" and cfg["backbone_params"]["main_backbone"].get(
                "freeze") else None,
            model.backbone.body.inter_depth if input_type == "Combined" and cfg["backbone_params"][
                "additional_backbone"].get("freeze") else None,
            model.backbone.body if input_type != "Combined" and cfg["backbone_params"]["main_backbone"].get(
                "freeze") else None,
            model.backbone.fpn if cfg["backbone_params"].get("freeze_fpn") else None,
            model.rpn if cfg["anchor_generator_params"].get("freeze") else None,
        )

        return model


"""
registry maps names of models to functions that return the models. The function
takes the number of classes and a bool stating if the model
should be pretrained. function returns a torch model object.

Please note, that any model that is meant to be used with the train_model framework
must implement the pytorch.nn.Module api, as well as having two extra methods:
    - train: puts the model in a state to train, calling the module will return losses
             against the given inputs.
    - eval: puts the model in a state of validation, calling the module will return an
            inference relative to the inputs.
Failing to implement these methods will cause potentail errors while training.
"""

models_registry = {
    "maskrcnn_resnet_base": MaskRCNNInitNamespace.get_maskrcnn_resnet_model,
}

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}
