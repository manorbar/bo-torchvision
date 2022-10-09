import os
import time
import pathlib
import pickle
import yaml
import sys
import cv2
import argparse

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

from torchvision.training.utils import get_grads, vis_grads, numpy2torch_image, put_text, torch2numpy_image, \
    create_image_mask_overlay_inference, load_state_dict_layer_by_layer
from torchvision.training.building_transform import BuildingTransform
from torchvision.training.building_dataset import dataset_registry
from torchvision.training.building_models import models_registry as models_reg


def parse_args():
    parser = argparse.ArgumentParser("Mask RCNN running module")
    parser.add_argument('-c', "--cfg", required=True, help="The configuration .yaml file path")
    parser.add_argument('-m', "--mode", required=True, help="The runtime mode: train or test")
    return vars(parser.parse_args())


def assert_args(args):
    if not os.path.isfile(args["cfg"]):
        raise Exception("Invalid config .yaml file")
    if args["mode"].lower() not in {"train", "test"}:
        raise Exception("The chosen mode is invalid. Choose either train or test")


def convert_buildings_data_to_mrcnn(data, device, train_mode=True):
    cp_data = dict()

    for k in data.keys():
        if train_mode:
            if k == "targets":
                cp_data[k] = [{k_: v.to(device) for k_, v in targets.items() if k_ != "image_id"} for targets in
                              data[k]]
            else:
                cp_data[k] = [image.to(device) for image in data[k]]
        elif k != "targets":
            cp_data[k] = [image.to(device) for image in data[k]]

    return cp_data


def update_tb_epoch_train_results(summary_writer, results_dict):
    if summary_writer:
        summary_writer.add_scalar("Loss/train_mean_loss", results_dict["mean_loss"], results_dict["epoch"])
        for loss_type, loss_value in results_dict["losses_sums"].items():
            summary_writer.add_scalar(f"Train_Losses/{loss_type}", loss_value, results_dict["epoch"])
        summary_writer.add_scalar("Optimizer/lr", results_dict["opt_state"]['param_groups'][0]['lr'],
                                  results_dict["epoch"])
        summary_writer.add_scalar("Optimizer/momentum", results_dict["opt_state"]['param_groups'][0]['momentum'],
                                  results_dict["epoch"])
        summary_writer.add_scalar("Optimizer/weight_decay",
                                  results_dict["opt_state"]['param_groups'][0]['weight_decay'], results_dict["epoch"])
        for name, tensor in results_dict["model_named_params"]:  # Layer histograms
            summary_writer.add_histogram(f"Parameters/{name}", tensor, results_dict["epoch"])


def update_tb_epoch_val_results(summary_writer, results_dict):
    if summary_writer:
        summary_writer.add_scalar("Loss/validation_mean_loss", results_dict["mean_loss"], results_dict["epoch"])
        for loss_type, loss_value in results_dict["losses_sums"].items():
            summary_writer.add_scalar(f"Validation_Losses/{loss_type}", loss_value, results_dict["epoch"])
        image_grid = make_grid(results_dict["vis_images"])
        summary_writer.add_image("Masks/validation_eval", image_grid, results_dict["epoch"])


def update_tb_grads_visualization(vis_grads_freq, i, model, grads, summary_writer, epoch):
    if vis_grads_freq is not None:
        if i % vis_grads_freq == 0:
            grads.append(get_grads(model))
        if len(grads) == 2:
            grads_graph = vis_grads(grads)
            summary_writer.add_image(f"Gradients/batch_{i - vis_grads_freq}-{i}", grads_graph, epoch)
            del grads[0]


def update_tb_highest_loss_images(model, loss, max_loss, epoch, data, sample_name, highest_loss_samples_info_lst,
                                  summary_writer, mask_conf_thresh=0.5, highest_loss_vis_thresh=2):
    if max_loss >= loss or loss < highest_loss_vis_thresh:
        return max_loss

    highest_loss_samples_info_lst.append({"sample_name": sample_name, "loss": loss})
    model.eval()
    pred = model(**data)
    pred = [{k: torch.Tensor.cpu(v) for k, v in p.items()} for p in pred][0]
    loss_txt = f"Loss: {loss}"

    # Bounding boxes
    img2display_boxes = numpy2torch_image(
        cv2.cvtColor((put_text(torch2numpy_image(data["main_input"][0]), loss_txt)), cv2.COLOR_BGR2RGB))
    summary_writer.add_image_with_boxes("Highest_Validation_Loss/boxes", img2display_boxes, pred["boxes"],
                                        global_step=epoch)
    summary_writer.add_image_with_boxes("Highest_Validation_Loss/gt_boxes", img2display_boxes,
                                        data["targets"][0]["boxes"], global_step=epoch)

    # Masks
    masked_img = numpy2torch_image(cv2.cvtColor(put_text(torch2numpy_image(
        create_image_mask_overlay_inference(data["main_input"][0], pred["masks"], alpha=0.5, thresh=mask_conf_thresh)),
        loss_txt), cv2.COLOR_BGR2RGB))
    gt_masked_img = numpy2torch_image(cv2.cvtColor(put_text(torch2numpy_image(
        create_image_mask_overlay_inference(data["main_input"][0], data["targets"][0]["masks"].unsqueeze(1).cpu(),
                                            alpha=0.5, thresh=mask_conf_thresh)), loss_txt), cv2.COLOR_BGR2RGB))
    summary_writer.add_image("Highest_Validation_Loss/masks", masked_img, global_step=epoch)
    summary_writer.add_image("Highest_Validation_Loss/gt_masks", gt_masked_img, global_step=epoch)

    return loss


def resume(resume_config, model, optimizer=None, scheduler=None):
    """
    resume to model position with optim state, model state, epoch and loss, if optimizer is given return an optimizer if not return None
    use the convention specified at: //https://pytorch.org/tutorials/beginner/saving_loading_models.html
    :param resume_config: resume dict configurations
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer, if not given return None
    :param scheduler: torch.optim.lr_scheduler, if not given return None
    :return: str, str, str, torch.nn.Module, torch.optim.Optimizer, int
    """

    summary_dir = os.path.join(resume_config['run_dir'], "summary")
    tb_dir = os.path.join(resume_config['run_dir'], "tensorboard")
    checkpoint_dir = os.path.join(resume_config['run_dir'], "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "last_epoch", "maskrcnn_resnet_base")

    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "rb") as in_checkpoint:
            checkpoint = torch.load(in_checkpoint)
            model = load_state_dict_layer_by_layer(model, checkpoint['model_state_dict'])
            if optimizer is not None and resume_config.get("resume_optim"):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and resume_config.get("resume_scheduler"):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return summary_dir, checkpoint_dir, tb_dir, model, optimizer, scheduler, int(checkpoint['epoch']) + 1

    raise FileNotFoundError(f"Invalid resume checkpoint path {checkpoint_path}")


def save_model(model, optimizer, lr_scheduler, epoch, loss, config, directory):
    """
    saves a serialized dictionary with the model state, optimizer
    state, epoch and loss.
    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :param optimizer: pytorch lr_scheduler
    :param epoch: int, current epoch
    :param loss: float, losses
    :param config: dict representing the config json.
    :param directory: string, path to directory.
    """
    # save checkpoint
    model_name = config['model']
    checkpoint_path = os.path.join(directory, model_name)
    print("saving model to: {}, epoch = {} ".format(checkpoint_path, epoch))
    with open(checkpoint_path, "wb") as out_model:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "loss": loss  # currently not in use
        }, out_model)
    return checkpoint_path


def output_dict_to_summary_writer(output_dict, summary_writer, epoch):
    """
    takes output from soutput dict and write to summary writer
    :param output_dict: dict, outputs calculated from calculate_statistics
    :param summary_writer: torch.utils.SummaryWriter, used to write to tensor board
    :param epoch: int, epoch to extract from output dict
    """
    tag_keys = output_dict[epoch].keys()
    for tag in tag_keys:

        beta_values = output_dict[epoch][tag].keys()
        for beta in beta_values:

            for scalar_key, scalar_value in output_dict[epoch][tag][beta].items():
                summary_writer.add_scalar(f"{tag}_{beta}/{scalar_key}", scalar_value, int(epoch))


def get_input_type(cfg):
    err_flag = False
    if not isinstance(cfg.get("backbone_params"), dict):
        err_flag = True
    backbone_params = cfg.get("backbone_params")
    assert isinstance(backbone_params, dict), f"Invalid backbone_params={backbone_params}"
    if not isinstance(backbone_params.get("main_backbone"), dict):
        err_flag = True
    if err_flag:
        raise Exception(f"Invalid backbone_params={cfg.get('backbone_params')}")

    if isinstance(backbone_params.get("additional_backbone"), dict) \
            and isinstance(backbone_params.get("fusion_type"), str):
        return "Combined"

    backbone_params = backbone_params.get("main_backbone")
    if "rgbd" in str(backbone_params.get("name")):
        return "RGBD"
    elif "depth" in str(backbone_params.get("name")):
        return "Depth"
    elif "resnet" in str(backbone_params.get("name")):  # TODO: Change to regex
        return "RGB"
    else:
        raise Exception(f"Invalid backbone name: {backbone_params.get('name')}")


def load_datasets(config):
    """
    creates transforms based on the parameters specified in the config file,
    and the corresponding datasets. parameters should be validated inside the
    dataset class.
    :param config: dict representing the config json.
    :return: torch.utils.data.Dataset: train, torch.utils.data.Dataset: validation
    """
    # create transforms
    train_transform = BuildingTransform(config['transform'], train=True)
    valid_transform = BuildingTransform(config['transform'], train=False)

    # load datasets
    train_ds = dataset_registry[config['dataset']](input_type=get_input_type(config),
                                                   trans=train_transform, **config['dataset_params']['train'])
    valid_ds = dataset_registry[config['dataset']](input_type=get_input_type(config),
                                                   trans=valid_transform, **config['dataset_params']['validation'])
    return train_ds, valid_ds


def get_model(config):
    """
    uses the geter functions in the registry to create a model.
    see building_models for requirements on this function.
    :param config: dict representing the config json.
    :return: torch.nn model.
    """
    model_name = config['model']
    # make sure model is registered
    if model_name not in models_reg:
        raise Exception("given model name is not registered in the building_models model registry.")

    model = models_reg[model_name](config, get_input_type(config))
    return model


def setup_run(config, mode="train"):
    """
    creates the logging directory if it doesn't exist yet,
    and then inside it creates a run directory. The run directory
    has a sub-dir for summary, tensor board dir and for checkpoints.
    inside the checkpoints dir we create sub-dirs for model parameters and statistic logs.
    we also init a dict to save best f1 performances during training.
    :param config: dict representing the json config file.
    :param mode: string etiher train or test
    :return: str: run dir, str: summary dir, str: checkpoint dir
    """

    # create logdir if it doesn't exist
    print("Creating log and checkpoint directories.")
    current_run = "run_" + time.strftime("%m-%d_%H-%M-%S", time.localtime()) + "_" + config["experiment_name"]
    log_dir = config['train_params']['logdir'] if mode == "train" else config['test_params']['logdir']
    if not os.path.isdir(log_dir):
        pathlib.Path(log_dir).mkdir(parents=True)

    # create specific run log directory
    run_log_dir = os.path.join(log_dir, current_run)
    pathlib.Path(run_log_dir).mkdir(parents=False)

    # create checkpoint dir and it sub-dirs
    checkpoint_dir = None
    if mode == "train":
        checkpoint_dir = os.path.join(run_log_dir, "checkpoints")
        pathlib.Path(checkpoint_dir).mkdir(parents=True)

        checkpoint_statistic_log_dir = os.path.join(checkpoint_dir, "statistic_logs")
        pathlib.Path(checkpoint_statistic_log_dir).mkdir(parents=True)

        # dirs for analysis of detection only
        checkpoint_last_epoch_dir = os.path.join(checkpoint_dir, "last_epoch")
        pathlib.Path(checkpoint_last_epoch_dir).mkdir(parents=True)

        checkpoint_best_total_dir = os.path.join(checkpoint_dir, "best_total")  # fixed f1
        pathlib.Path(checkpoint_best_total_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_small")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_medium_dir = os.path.join(checkpoint_dir, "best_medium")
        pathlib.Path(checkpoint_best_medium_dir).mkdir(parents=True)

        checkpoint_best_large_dir = os.path.join(checkpoint_dir, "best_large")
        pathlib.Path(checkpoint_best_large_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_map")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_fixed_map")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_f1")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_f1_small")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_f1_medium")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_f1_large")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        # dirs for analysis including classification task
        checkpoint_best_total_dir = os.path.join(checkpoint_dir, "best_total_cls")  # fixed f1
        pathlib.Path(checkpoint_best_total_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_small_cls")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_medium_dir = os.path.join(checkpoint_dir, "best_medium_cls")
        pathlib.Path(checkpoint_best_medium_dir).mkdir(parents=True)

        checkpoint_best_large_dir = os.path.join(checkpoint_dir, "best_large_cls")
        pathlib.Path(checkpoint_best_large_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_map_cls")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_fixed_map_cls")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

        checkpoint_best_small_dir = os.path.join(checkpoint_dir, "best_f1_cls")
        pathlib.Path(checkpoint_best_small_dir).mkdir(parents=True)

    # save config file to checkpoint folder
    with open(os.path.join(checkpoint_dir, "run_config.yaml"), 'w') as cfg_file:
        yaml.dump(config, cfg_file, default_flow_style=False)

    # init dict to save best f1 results
    result_dict = {}

    result_dict['best_total'] = 0
    result_dict['best_total_epoch'] = 0

    result_dict['best_small'] = 0
    result_dict['best_small_epoch'] = 0

    result_dict['best_medium'] = 0
    result_dict['best_medium_epoch'] = 0

    result_dict['best_large'] = 0
    result_dict['best_large_epoch'] = 0

    result_dict['best_f1'] = 0
    result_dict['best_f1_epoch'] = 0

    result_dict['best_f1_small'] = 0
    result_dict['best_f1_small_epoch'] = 0

    result_dict['best_f1_medium'] = 0
    result_dict['best_f1_medium_epoch'] = 0

    result_dict['best_f1_large'] = 0
    result_dict['best_f1_large_epoch'] = 0

    result_dict['best_map'] = 0
    result_dict['best_map_epoch'] = 0

    result_dict['best_fixed_map'] = 0
    result_dict['best_fixed_map_epoch'] = 0

    # including classification part
    result_dict['best_total_cls'] = 0
    result_dict['best_total_cls_epoch'] = 0

    result_dict['best_small_cls'] = 0
    result_dict['best_small_cls_epoch'] = 0

    result_dict['best_medium_cls'] = 0
    result_dict['best_medium_cls_epoch'] = 0

    result_dict['best_large_cls'] = 0
    result_dict['best_large_cls_epoch'] = 0

    result_dict['best_f1_cls'] = 0
    result_dict['best_f1_cls_epoch'] = 0

    result_dict['best_map_cls'] = 0
    result_dict['best_map_cls_epoch'] = 0

    result_dict['best_fixed_map_cls'] = 0
    result_dict['best_fixed_map_cls_epoch'] = 0

    with open(os.path.join(checkpoint_dir, "result_dict.pkl"), 'wb') as f:
        pickle.dump(result_dict, f)

    # create summary dir for images
    summary_dir = os.path.join(run_log_dir, "summary")
    pathlib.Path(summary_dir).mkdir(parents=True)

    # create tensorboard directory
    tb_dir = os.path.join(run_log_dir, "tensorboard")
    pathlib.Path(tb_dir).mkdir(parents=True)

    return run_log_dir, summary_dir, checkpoint_dir, tb_dir


# def resume_setup(run_log_dir):
#     """
#     this function is used when beck to training after it stop. it assumed the dir and files are already exists.
#     :param run_log_dir: path to the train log dir
#     :return: restore paths to all training directories
#     """
#
#     summary_dir = os.path.join(run_log_dir, "summary")
#     tb_dir = os.path.join(run_log_dir, "tensorboard")
#     checkpoint_dir = os.path.join(run_log_dir, "checkpoints")
#
#     return run_log_dir, summary_dir, checkpoint_dir, tb_dir


def get_optimizer(params, config):
    """
    names of the parameters in the optimizer_params section of
    the configuration file must match the names of the arguments
    of the chosen optimizer, see the pytorch optim library documentation.
    :param params: optmizer params
    :param config: dict representing the configuration json.
    :return: torch.optim.Optimizer
    """
    if config['optimizer'] not in {"SGD", "Adam", "Adagrad"}:
        raise Exception("{} is not a supported optimizer.".format(config['optimizer']))
    else:
        if config['optimizer'] == "SGD":
            return optim.SGD(params, **config['optimizer_params'])
        elif config['optimizer'] == "Adam":
            return optim.Adam(params, **config['optimizer_params'])
        else:
            return optim.Adagrad(params, **config['optimizer_params'])


def get_lr_scheduler(optimizer, config):
    if config['lr_scheduler'] not in {"ReduceLROnPlateau"}:
        raise Exception("{} is not a supported scheduler.".format(config['lr_scheduler']))
    else:
        if config['lr_scheduler'] == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, **config['lr_scheduler_params'])
        else:
            return None
