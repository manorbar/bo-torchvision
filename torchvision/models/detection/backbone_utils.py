import warnings
from typing import Callable, Dict, List, Optional, Union

from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from .. import mobilenet, resnet
from .._api import _get_enum_from_fn, WeightsEnum
from .._utils import handle_legacy_interface, IntermediateLayerGetter


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
            self,
            backbone: nn.Module,
            return_layers: Dict[str, str],
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNGivenBody(nn.Sequential):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
	@@ -25,218 +31,315 @@ class BackboneWithFPN(nn.Module):
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, body, in_channels_list, out_channels, two_sides_fpn=False):

        if two_sides_fpn:
            fpn = TwoSidesFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        else:
            fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        super(BackboneWithFPNGivenBody, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))


@handle_legacy_interface(
    weights=(
            "pretrained",
            lambda kwargs: _get_enum_from_fn(resnet.__dict__[kwargs["backbone_name"]]).from_str("IMAGENET1K_V1"),
    ),
)
def resnet_fpn_backbone(
        *,
        backbone_name: str,
        weights: Optional[WeightsEnum],
        norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers: int = 3,
        returned_layers: Optional[List[int]] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Args:
        backbone_name (string): resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        weights (WeightsEnum, optional): The pretrained weights for the model
        norm_layer (callable): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default a ``LastLevelMaxPool`` is used.
    """
    backbone = resnet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks)


def _resnet_fpn_extractor(
        backbone: resnet.ResNet,
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )


def _validate_trainable_layers(
        is_trained: bool,
        trainable_backbone_layers: Optional[int],
        max_value: int,
        default_value: int,
) -> int:
    # don't freeze any layers if pretrained model or backbone is not used
    if not is_trained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                f"falling back to trainable_backbone_layers={max_value} so that all layers are trainable"
            )
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    if trainable_backbone_layers < 0 or trainable_backbone_layers > max_value:
        raise ValueError(
            f"Trainable backbone layers should be in the range [0,{max_value}], got {trainable_backbone_layers} "
        )
    return trainable_backbone_layers


@handle_legacy_interface(
    weights=(
            "pretrained",
            lambda kwargs: _get_enum_from_fn(mobilenet.__dict__[kwargs["backbone_name"]]).from_str("IMAGENET1K_V1"),
    ),
)
def mobilenet_backbone(
        *,
        backbone_name: str,
        weights: Optional[WeightsEnum],
        fpn: bool,
        norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers: int = 2,
        returned_layers: Optional[List[int]] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
) -> nn.Module:
    backbone = mobilenet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _mobilenet_extractor(backbone, fpn, trainable_layers, returned_layers, extra_blocks)


def _mobilenet_extractor(
        backbone: Union[mobilenet.MobileNetV2, mobilenet.MobileNetV3],
        fpn: bool,
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
            raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")
        return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        return BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        )
    else:
        m = nn.Sequential(
            backbone,
            # depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels  # type: ignore[assignment]
        return m


class BackboneWithFPN(BackboneWithFPNGivenBody):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, two_sides_fpn=False):
        super().__init__(IntermediateLayerGetter(backbone, return_layers=return_layers), in_channels_list, out_channels,
                         two_sides_fpn)


class BackboneWithLateFusionFPN(BackboneWithFPNGivenBody):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone1, backbone2, return_layers, in_channels_list, out_channels, two_sides_fpn=False,
                 fusion_type="concat"):
        global fusion_types
        assert fusion_type in fusion_types, f"Invalid fusion type {fusion_type}. Valid fusion types: {fusion_types}"

        body = None
        if fusion_type == "Concat":
            body = IntermediateLayerGetterLateFusionConcat(backbone1, backbone2, return_layers=return_layers)

        elif fusion_type == "Sum":
            body = IntermediateLayerGetterLateFusionSummation(backbone1, backbone2, return_layers=return_layers)

        elif fusion_type == "Attention":
            body = IntermediateLayerGetterLateAttentionFusion(backbone1, backbone2, return_layers, in_channels_list)

        super().__init__(body, in_channels_list, out_channels, two_sides_fpn)


class ResnetFPNNamespace:

    @staticmethod
    def assert_single_backbone_params(backbone_name, pretrained, backbone_names=resnet.__all__):
        assert isinstance(backbone_name, str) and backbone_name in backbone_names, \
            f"Invalid argument backbone_name={backbone_name}. Valid backbone names are {backbone_names}"
        assert isinstance(pretrained, str) and os.path.exists(str(pretrained)) or isinstance(pretrained, bool), \
            f"Invalid argument pretrained={pretrained}. 'pretrained' should be a valid path or a bool flag"

    @staticmethod
    def assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn):
        assert isinstance(num_of_layers_in_pyramid, int), \
            f"Invalid argument num_of_layers_in_pyramid={num_of_layers_in_pyramid}. 'num_of_layers_in_pyramid' should be an integer"
        assert isinstance(two_sides_fpn, bool), \
            f"Invalid argument two_sides_fpn={two_sides_fpn}. 'two_sides_fpn' should be a boolean"

    @staticmethod
    def resnet_fpn_backbone(backbone1=None, backbone2=None, fusion_type=None, num_of_layers_in_pyramid=4,
                            two_sides_fpn=False):
        assert backbone1 or backbone2, "Invalid arguments. 'resnet_fpn_backbone' should get at least one backbone " \
                                       "argument backbone1={backbone1}, backbone2={backbone2}"

        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        return_layers = {k: v for k, v in zip(list(return_layers.keys())[:num_of_layers_in_pyramid],
                                              list(return_layers.values())[:num_of_layers_in_pyramid])}

        fpn_in_channels = backbone1.inplanes // (4 if fusion_type == "Concat" else 8)
        in_channels_list = [
            fpn_in_channels,
            fpn_in_channels * 2,
            fpn_in_channels * 4,
            fpn_in_channels * 8,
        ]
        out_channels = 256

        if backbone1 and backbone2 and fusion_type:
            return BackboneWithLateFusionFPN(
                backbone1,
                backbone2,
                return_layers,
                in_channels_list[:num_of_layers_in_pyramid],
                out_channels,
                two_sides_fpn,
                fusion_type
            )

        return BackboneWithFPN(
            backbone1 if backbone1 else backbone2,
            return_layers,
            in_channels_list[:num_of_layers_in_pyramid],
            out_channels,
            two_sides_fpn
        )

    class SingleBackboneNamespace:

        @staticmethod
        def assert_backbone_params(backbone_name, pretrained, num_of_layers_in_pyramid, two_sides_fpn,
                                   valid_backbone_names):
            ResnetFPNNamespace.assert_single_backbone_params(backbone_name, pretrained, valid_backbone_names)
            ResnetFPNNamespace.assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn)

        @staticmethod
        def resnet_fpn_single_input_backbone(backbone_name="resnet50", pretrained=False, num_of_layers_in_pyramid=4,
                                             two_sides_fpn=False, valid_backbone_names=resnet.__all__):
            ResnetFPNNamespace.SingleBackboneNamespace.assert_backbone_params(backbone_name, pretrained,
                                                                              num_of_layers_in_pyramid, two_sides_fpn,
                                                                              valid_backbone_names)

            backbone = resnet.__dict__[backbone_name](
                pretrained=pretrained,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            return ResnetFPNNamespace.resnet_fpn_backbone(
                backbone1=backbone,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn
            )

        @staticmethod
        def resnet_rgb_fpn_backbone(backbone_name="resnet50", pretrained=False, num_of_layers_in_pyramid=4,
                                    two_sides_fpn=False):
            print("Creating RGB resnet FPN backbone")
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["RGB"]
            )

        @staticmethod
        def resnet_rgbd_fpn_backbone(backbone_name="resnet50_rgbd", pretrained=False, num_of_layers_in_pyramid=4,
                                     two_sides_fpn=False):
            print("Creating RGBD resnet FPN backbone")
            assert backbone_name in resnet.model_names["RGBD"], f"Invalid backbone name for function" \
                                                                f" {ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgbd_fpn_backbone.__name__}." \
                                                                f" Valid names: {resnet.model_names['RGBD']}"
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["RGBD"]
            )

        @staticmethod
        def resnet_depth_fpn_backbone(backbone_name="resnet50_depth", pretrained=False, num_of_layers_in_pyramid=4,
                                      two_sides_fpn=False):
            print("Creating Depth resnet FPN backbone")
            assert backbone_name in resnet.model_names["Depth"], f"Invalid backbone name for function" \
                                                                 f" {ResnetFPNNamespace.SingleBackboneNamespace.resnet_depth_fpn_backbone.__name__}." \
                                                                 f" Valid names: {resnet.model_names['Depth']}"
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["Depth"]
            )

    class DoubleBackboneNamespace:

        @staticmethod
        def assert_backbone_params(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid, two_sides_fpn,
                                   fusion_type):
            global fusion_types
            ResnetFPNNamespace.assert_single_backbone_params(rgb_backbone_params["name"],
                                                             rgb_backbone_params["pretrained"],
                                                             resnet.model_names["RGB"])
            ResnetFPNNamespace.assert_single_backbone_params(depth_backbone_params["name"],
                                                             depth_backbone_params["pretrained"],
                                                             resnet.model_names["Depth"])
            ResnetFPNNamespace.assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn)
            assert fusion_type in fusion_types, f"Invalid fusion_type={fusion_type}, Valid fusion types: {fusion_types}"

        @staticmethod
        def resnet_late_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                            two_sides_fpn=False, fusion_type="Sum"):
            ResnetFPNNamespace.DoubleBackboneNamespace.assert_backbone_params(rgb_backbone_params,
                                                                              depth_backbone_params,
                                                                              num_of_layers_in_pyramid, two_sides_fpn,
                                                                              fusion_type)
            print(f"Creating late fusion resnet FPN backbone with '{fusion_type}' fusion type")

            rgb_backbone = resnet.__dict__[rgb_backbone_params["name"]](
                pretrained=rgb_backbone_params["pretrained"],
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            depth_backbone = resnet.__dict__[depth_backbone_params["name"]](
                pretrained=depth_backbone_params["pretrained"],
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            return ResnetFPNNamespace.resnet_fpn_backbone(
                backbone1=rgb_backbone,
                backbone2=depth_backbone,
                fusion_type=fusion_type,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn
            )

        @staticmethod
        def resnet_sum_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                           two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Sum"
            )

        @staticmethod
        def resnet_concat_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                              two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Concat"
            )

        @staticmethod
        def resnet_attention_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                                 two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Attention"
            )
