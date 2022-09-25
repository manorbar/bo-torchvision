from torchvision.models.detection.backbone_utils import ResnetFPNNamespace

fpn_registry = {
    "RGB": ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgb_fpn_backbone,
    "Depth": ResnetFPNNamespace.SingleBackboneNamespace.resnet_depth_fpn_backbone,
    "RGBD": ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgbd_fpn_backbone,
    "Combined": {
        "Sum": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_sum_fusion_fpn_backbone,
        "Concat": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_concat_fusion_fpn_backbone,
        "Attention": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_attention_fusion_fpn_backbone,
    }
}

input_types = {"RGB", "RGBD", "Depth", "Combined"}
fusion_types = {"Sum", "Concat", "Attention"}


def fpn_factory(backbone_params, input_type, fusion_type):
    global fpn_registry
    assert input_type in input_types, f"Invalid input type {input_type}. Valid input types: {input_types}"
    if input_type == "RGB":
        return fpn_registry["RGB"](**backbone_params)
    elif input_type == "RGBD":
        return fpn_registry["RGBD"](**backbone_params)
    elif input_type == "Depth":
        return fpn_registry["Depth"](**backbone_params)
    elif input_type == "Combined":
        return fpn_registry["Combined"][fusion_type](**backbone_params)
