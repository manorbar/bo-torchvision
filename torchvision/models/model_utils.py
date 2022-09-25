import torch
from collections import OrderedDict
from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class RGBDAttentionFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(RGBDAttentionFusionBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.q = nn.Conv2d(in_channels, in_channels, (1, 1), bias=False)  # f_rgb_attention
        self.k = nn.Conv2d(in_channels, in_channels, (1, 1), bias=False)  # f_depth_attention
        self.v = nn.Conv2d(in_channels, in_channels, (1, 1), bias=False)  # f_rgb

    def rgbd_attention(self, rgb_queries, depth_keys, rgb_values):
        """
        Args:
            rgb_queries: torch.tensor in the shape of (B, C, H, W)
            depth_keys: torch.tensor in the shape of (B, C, H, W)
            rgb_values: torch.tensor in the shape of (B, C, H, W)
        Returns:
            torch.tensor as the attention result (B, C, H, W)
        """
        b, c, h, w = rgb_values.shape
        rgb_queries = self.avg(rgb_queries)  # (B, C, 1, 1)
        depth_keys = self.avg(depth_keys)  # (B, C, 1, 1)
        attention_weights = (rgb_queries * depth_keys).softmax(dim=1)  # (B, C, 1, 1)
        # Expand attention_weights to shape (B, C, H, W) in order to do batch element-wise matrix multiplication with rgb_values
        expanded_attention_weights = attention_weights.expand((b, c, h, w))  # (B, C, H, W)
        return rgb_values * expanded_attention_weights  # (B, C, H, W)

    def forward(self, rgb, depth):
        """
        Args:
            self:
            rgb: rgb feature map (torch.tensor) with the shape of (B, C, H, W)
            depth: depth feature map (torch.tensor) with the shape of (B, C, H, W)
        Returns:
            RGB-D fused attention result and weights (torch.tensor ,torch.tensor)
            with the shapes of (B, C, H, W), (B, C, 1, 1)
        """
        assert rgb.shape == depth.shape, "Invalid shapes. Expected rgb.shape to be equals to depth.shape," + \
                                         f" but rgb.shape={rgb.shape} and depth.shape={depth.shape}"
        return rgb + self.rgbd_attention(self.q(rgb), self.k(depth), self.v(rgb))


class IntermediateLayerGetterLateFusion(nn.Module):
    def __init__(self, rgb_backbone, depth_backbone, return_layers):
        super().__init__()
        self.inter_rgb = IntermediateLayerGetter(rgb_backbone, return_layers=return_layers)
        self.inter_depth = IntermediateLayerGetter(depth_backbone, return_layers=return_layers)

    def _forward(self, x, feature_maps_fusions):
        xrgb, xd = x
        rgb_features = self.inter_rgb(xrgb)
        depth_features = self.inter_depth(xd)
        out = OrderedDict()
        for (rgb_fm_key, rgb_fm_val), (_, d_fm_val), fusion in zip(rgb_features.items(), depth_features.items(),
                                                                   feature_maps_fusions):
            out[rgb_fm_key] = fusion(rgb_fm_val, d_fm_val)
        return out

    def forward(self, x):
        raise NotImplementedError


class IntermediateLayerGetterLateFusionConcat(IntermediateLayerGetterLateFusion):
    def __init__(self, rgb_backbone, depth_backbone, return_layers):
        super().__init__(rgb_backbone, depth_backbone, return_layers)
        self.return_layers = return_layers

    def forward(self, x):
        return super()._forward(x, [lambda rgb, d: torch.cat([rgb, d], dim=1)] * len(self.return_layers))


class IntermediateLayerGetterLateFusionSummation(IntermediateLayerGetterLateFusion):
    def __init__(self, rgb_backbone, depth_backbone, return_layers):
        super().__init__(rgb_backbone, depth_backbone, return_layers)
        self.return_layers = return_layers

    def forward(self, x):
        return super()._forward(x, [lambda rgb, d: rgb + d] * len(self.return_layers))


class IntermediateLayerGetterLateAttentionFusion(IntermediateLayerGetterLateFusion):
    def __init__(self, rgb_backbone, depth_backbone, return_layers, out_channels_list):
        super().__init__(rgb_backbone, depth_backbone, return_layers)
        self.rgbd_attention_blocks = nn.ModuleList([RGBDAttentionFusionBlock(oc) for oc in out_channels_list])

    def forward(self, x):
        return super()._forward(x, self.rgbd_attention_blocks)
