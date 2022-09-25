from copy import deepcopy


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
