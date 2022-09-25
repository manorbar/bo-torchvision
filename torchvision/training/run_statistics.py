import os
import pickle
import sys
from runtime_utils import save_model
from torchvision.models.detection.mask_rcnn import MaskRCNN
from building_models import models_registry
import torch
from torch.utils.data import DataLoader
from building_dataset import dataset_registry
from building_transform import BuildingTransform
from building_statistics_utils import MaskStatisticsTorch
from utils import create_empty_dict_key_if_required, collate_dict_fn
from runtime_utils import get_input_type
import numpy as np


def eval_statistics_wrapper(checkpoint_dir, stats_pickle_path, config, epoch, optimizer, lr_scheduler, current_mean_loss, model=None, model_state_dict=None):
    """
    wrapper for running eval statistics
    :param checkpoint_dir: str, path to checkpoint file
    :param stats_pickle_path: str, path to pickle file
    :param config: dict, config file
    :param epoch: int, current epoch
    :param optimizer: model optimizer
    :param lr_scheduler: model lr_scheduler
    :param current_mean_loss: float, loss
    :param model: maskrcnn model/None
    :param model_state_dict: dict, model param to reconstruct him
    :param model: nn.module, if not given will create according to checkpoint
    """

    # set stdout to be the log file
    checkpoint_statistic_log_dir = os.path.join(checkpoint_dir, "statistic_logs")
    original_std_out = sys.stdout
    sys.stdout = open(checkpoint_statistic_log_dir + '/statistic_logs_epoch_{}.txt'.format(epoch), 'w')

    # create datasets
    valid_transform = BuildingTransform(config['transform'], train=False)
    valid_ds = dataset_registry[config['dataset']](input_type=get_input_type(config),
        trans=valid_transform, **config['dataset_params']['validation'])
    valid_loader = DataLoader(valid_ds, collate_fn=collate_dict_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # undo to the pop operation in model registry
    config['general_model_params']['pretrained'] = True

    if model is None:
        # build model from checkpoint
        model: MaskRCNN = models_registry[config['model']](config).to(device)
        model.load_state_dict(model_state_dict)

    # evaluate statistics changing output dict
    output_dict = eval_statistics(model, valid_loader, device, epoch, config)

    with open(os.path.join(stats_pickle_path, f"stats_epoch_{epoch}.p"), 'wb') as pickle_handle:
        pickle.dump(output_dict, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving the model parameters in the required fields:

    with open(os.path.join(checkpoint_dir, "result_dict.pkl"), 'rb') as f:
        result_dict = pickle.load(f)

    for key, value in zip(list(result_dict.keys()), list(result_dict.values())):
        if type(value) == np.float64:
            print(f'{key}\t{value:.4f}')
        else:
            print(f'{key}\t{value}')

    _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "last_epoch"))

    # detection part
    if output_dict[epoch]["Total_statistics_detection"][1]["fixed_f1"] > result_dict['best_total']:

        print('best total =  {}'.format(result_dict['best_total']))
        print('validation_fixed_f1 = {}'.format(output_dict[epoch]["Total_statistics_detection"][1]["fixed_f1"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir,  "best_total"))
        result_dict['best_total'] = output_dict[epoch]["Total_statistics_detection"][1]["fixed_f1"]
        result_dict['best_total_epoch'] = epoch

    if output_dict[epoch]["Total_statistics_detection"][1]["f1"] > result_dict['best_f1']:
        print('best_f1 =  {}'.format(result_dict['best_f1']))
        print('f1 = {}'.format(output_dict[epoch]["Total_statistics_detection"][1]["f1"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_f1"))
        result_dict['best_f1'] = output_dict[epoch]["Total_statistics_detection"][1]["f1"]
        result_dict['best_f1_epoch'] = epoch

    if output_dict[epoch]["Total_statistics_detection"][1]["mAP"] > result_dict['best_map']:
        print('best_map =  {}'.format(result_dict['best_map']))
        print('map = {}'.format(output_dict[epoch]["Total_statistics_detection"][1]["mAP"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_map"))
        result_dict['best_map'] = output_dict[epoch]["Total_statistics_detection"][1]["mAP"]
        result_dict['best_map_epoch'] = epoch

    if output_dict[epoch]["Total_statistics_detection"][1]["fixed_mAP"] > result_dict['best_fixed_map']:
        print('best_fixed_map =  {}'.format(result_dict['best_fixed_map']))
        print('fixed_map = {}'.format(output_dict[epoch]["Total_statistics_detection"][1]["fixed_mAP"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_fixed_map"))
        result_dict['best_fixed_map'] = output_dict[epoch]["Total_statistics_detection"][1]["fixed_mAP"]
        result_dict['best_fixed_map_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_small"] > result_dict['best_small']:

        print('best_small =  {}'.format(result_dict['best_small']))
        print('validation_fixed_f1_small = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_small"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_small"))
        result_dict['best_small'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_small"]
        result_dict['best_small_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_medium"] > result_dict['best_medium']:

        print('best_medium =  {}'.format(result_dict['best_medium']))
        print('validation_fixed_f1_medium = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_medium"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_medium"))
        result_dict['best_medium'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_medium"]
        result_dict['best_medium_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_large"] > result_dict['best_large']:

        print('best_large =  {}'.format(result_dict['best_large']))
        print('validation_fixed_f1_large = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_large"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_large"))
        result_dict['best_large'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_fixed_f1_large"]
        result_dict['best_large_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_small"] > result_dict['best_f1_small']:
        print('best_f1_small =  {}'.format(result_dict['best_f1_small']))
        print('f1_small = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_small"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_f1_small"))
        result_dict['best_f1_small'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_small"]
        result_dict['best_f1_small_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_medium"] > result_dict['best_f1_medium']:
        print('best_f1_medium =  {}'.format(result_dict['best_f1_medium']))
        print('f1_medium = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_medium"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_f1_medium"))
        result_dict['best_f1_medium'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_medium"]
        result_dict['best_f1_medium_epoch'] = epoch

    if output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_large"] > result_dict['best_f1_large']:
        print('best_f1_large =  {}'.format(result_dict['best_f1_large']))
        print('f1_large = {}'.format(output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_large"]))

        _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_f1_large"))
        result_dict['best_f1_large'] = output_dict[epoch]["Total_Statistics_detection_per_size"][1]["validation_f1_large"]
        result_dict['best_f1_large_epoch'] = epoch

    # including classification part:
    if config['num_classes']-1 > 1:

        if output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_f1"] > result_dict['best_total_cls']:
            print('best total cls =  {}'.format(result_dict['best_total_cls']))
            print('validation_fixed_f1 cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_f1"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_total_cls"))
            result_dict['best_total_cls'] = output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_f1"]
            result_dict['best_total_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification"][1]["f1"] > result_dict['best_f1_cls']:
            print('best_f1_cls =  {}'.format(result_dict['best_f1_cls']))
            print('f1_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification"][1]["f1"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_f1_cls"))
            result_dict['best_f1_cls'] = output_dict[epoch]["Total_statistics_with_classification"][1]["f1"]
            result_dict['best_f1_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification"][1]["mAP"] > result_dict['best_map_cls']:
            print('best_map_cls =  {}'.format(result_dict['best_map_cls']))
            print('map_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification"][1]["mAP"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_map_cls"))
            result_dict['best_map_cls'] = output_dict[epoch]["Total_statistics_with_classification"][1]["mAP"]
            result_dict['best_map_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_mAP"] > result_dict['best_fixed_map_cls']:
            print('best_fixed_map_cls =  {}'.format(result_dict['best_fixed_map_cls']))
            print('fixed_map_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_mAP"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_fixed_map_cls"))
            result_dict['best_fixed_map_cls'] = output_dict[epoch]["Total_statistics_with_classification"][1]["fixed_mAP"]
            result_dict['best_fixed_map_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_small"] > result_dict['best_small_cls']:
            print('best_small_cls =  {}'.format(result_dict['best_small_cls']))
            print('validation_fixed_f1_small_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_small"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_small_cls"))
            result_dict['best_small_cls'] = output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_small"]
            result_dict['best_small_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_medium"] > result_dict['best_medium_cls']:
            print('best_medium_cls =  {}'.format(result_dict['best_medium_cls']))
            print('validation_fixed_f1_medium_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_medium"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_medium_cls"))
            result_dict['best_medium_cls'] = output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_medium"]
            result_dict['best_medium_cls_epoch'] = epoch

        if output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_large"] > result_dict['best_large_cls']:
            print('best_large_cls =  {}'.format(result_dict['best_large_cls']))
            print('validation_fixed_f1_large_cls = {}'.format(output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_large"]))

            _ = save_model(model, optimizer, lr_scheduler, epoch, current_mean_loss, config, os.path.join(checkpoint_dir, "best_large_cls"))
            result_dict['best_large_cls'] = output_dict[epoch]["Total_statistics_with_classification_per_size"][1]["validation_fixed_f1_large"]
            result_dict['best_large_cls_epoch'] = epoch

    # save the updated results dict
    with open(os.path.join(checkpoint_dir, "result_dict.pkl"), 'wb') as f:
        pickle.dump(result_dict, f)

    sys.stdout = original_std_out


def eval_statistics(model, data_loader, device, epoch, config):
    """
    evaluate statistics on validation set including map, precision, recall etc.
    :param model: nn.model to evaluate
    :param data_loader: pytorch validation dataloader
    :param device: torch.device to evaluate on, should be torch.device('cpu') or 'cuda'
    :param epoch: int, current epoch
    :param config: dict, representing json config file
    :return: dict, output results
    """
    # Unpack config
    iou_threshold = config['validation_params']['map_iou_threshold']
    num_classes = config['num_classes']-1
    sizes_thresh = tuple(config['validation_params']['sizes_thresh'])

    model.to(device)
    model.eval()

    # change score thresh to lowest possible value
    temp_box_score_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    # calculate the mean average precision on the validation dataset
    mst = MaskStatisticsTorch(iou_threshold=iou_threshold, dataset=data_loader.dataset, model=model, device=device, classes=num_classes, size_thresh=sizes_thresh,
                              scaling_factor=config['validation_params']['scaling_factor'])

    # We calculate the best f1 beta = 1 , and save the results for the confidence with the highest score
    for beta in [1]:
        output_dict = calculate_statistics(mst, epoch, beta=beta)

    # return to original thresh
    model.roi_heads.score_thresh = temp_box_score_thresh

    return output_dict


def calculate_statistics(mst, epoch, beta=1):
    """
    :param mst: MaskStatistics object
    :param epoch: int, epoch num
    :param beta: float, beta to calc f1 score
    :return: dict, output dict with statistics results
    """

    output_dict = {}

    # Total detection statistic part
    mAP, best_f1, best_thresh_idx, best_pr_idx, precision_list, recall_list = mst.get_map_and_best_scores(fixed=False, include_classification=False)
    fix_mAP, fix_best_f1, fix_best_thresh_idx, fixed_best_pr_idx, fix_precision_list, fix_recall_list = mst.get_map_and_best_scores(include_classification=False)

    mean_iou = mst.get_mean_iou(mst.confidences[fix_best_thresh_idx])
    mean_hausdorff_distance = mst.get_mean_hausdorff_distance(mst.confidences[fix_best_thresh_idx])

    create_empty_dict_key_if_required(output_dict, epoch)

    if mAP != -1:
        create_empty_dict_key_if_required(output_dict[epoch], "Total_statistics_detection")
        create_empty_dict_key_if_required(output_dict[epoch]["Total_statistics_detection"], beta)

        output_dict[epoch]["Total_statistics_detection"][beta]["validation_precision"] = precision_list[best_pr_idx]
        output_dict[epoch]["Total_statistics_detection"][beta]["validation_recall"] = recall_list[best_pr_idx]

        output_dict[epoch]["Total_statistics_detection"][beta]["validation_fixed_precision"] = fix_precision_list[fixed_best_pr_idx]
        output_dict[epoch]["Total_statistics_detection"][beta]["validation_fixed_recall"] = fix_recall_list[fixed_best_pr_idx]

        output_dict[epoch]["Total_statistics_detection"][beta]["f1"] = best_f1
        output_dict[epoch]["Total_statistics_detection"][beta]["conf"] = mst.confidences[best_thresh_idx]

        output_dict[epoch]["Total_statistics_detection"][beta]["fixed_f1"] = fix_best_f1
        output_dict[epoch]["Total_statistics_detection"][beta]["fixed_conf"] = mst.confidences[fix_best_thresh_idx]

        output_dict[epoch]["Total_statistics_detection"][beta]["mAP"] = mAP
        output_dict[epoch]["Total_statistics_detection"][beta]["fixed_mAP"] = fix_mAP

        output_dict[epoch]["Total_statistics_detection"][beta]["mean_iou"] = mean_iou
        output_dict[epoch]["Total_statistics_detection"][beta]["mean_hausdorff_distance"] = mean_hausdorff_distance

    # total statistic by size
    for size, size_name in enumerate(['small', 'medium', 'large']):
        mAP, best_f1, _, _, precision_list, recall_list = mst.get_map_and_best_scores(size=size, fixed=False, include_classification=False)
        fix_mAP, fix_best_f1, _, _, fix_precision_list, fix_recall_list = mst.get_map_and_best_scores(size=size, include_classification=False)

        if mAP != -1:
            create_empty_dict_key_if_required(output_dict[epoch], "Total_Statistics_detection_per_size")
            create_empty_dict_key_if_required(output_dict[epoch]["Total_Statistics_detection_per_size"], beta)

            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_precision_{size_name}"] = precision_list[best_pr_idx]
            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_recall_{size_name}"] = recall_list[best_pr_idx]
            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_f1_{size_name}"] = best_f1

            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_fixed_precision_{size_name}"] = fix_precision_list[fixed_best_pr_idx]
            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_fixed_recall_{size_name}"] = fix_recall_list[fixed_best_pr_idx]
            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"validation_fixed_f1_{size_name}"] = fix_best_f1

            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"mAP_{size_name}"] = mAP
            output_dict[epoch]["Total_Statistics_detection_per_size"][beta][f"fixed_mAP_{size_name}"] = fix_mAP

    # Total detection+classification statistic part:

    if mst.classes > 1:
        mAP, best_f1, best_thresh_idx, best_pr_idx, precision_list, recall_list = mst.get_map_and_best_scores(fixed=False, include_classification=True)
        fix_mAP, fix_best_f1, fix_best_thresh_idx, fixed_best_pr_idx, fix_precision_list, fix_recall_list = mst.get_map_and_best_scores(include_classification=True)

        mean_iou = mst.get_mean_iou(mst.confidences[fix_best_thresh_idx])
        mean_hausdorff_distance = mst.get_mean_hausdorff_distance(mst.confidences[fix_best_thresh_idx])

        create_empty_dict_key_if_required(output_dict, epoch)

        if mAP != -1:
            create_empty_dict_key_if_required(output_dict[epoch], "Total_statistics_with_classification")
            create_empty_dict_key_if_required(output_dict[epoch]["Total_statistics_with_classification"], beta)

            output_dict[epoch]["Total_statistics_with_classification"][beta]["validation_precision"] = precision_list[best_pr_idx]
            output_dict[epoch]["Total_statistics_with_classification"][beta]["validation_recall"] = recall_list[best_pr_idx]

            output_dict[epoch]["Total_statistics_with_classification"][beta]["validation_fixed_precision"] = fix_precision_list[fixed_best_pr_idx]
            output_dict[epoch]["Total_statistics_with_classification"][beta]["validation_fixed_recall"] = fix_recall_list[fixed_best_pr_idx]

            output_dict[epoch]["Total_statistics_with_classification"][beta]["f1"] = best_f1
            output_dict[epoch]["Total_statistics_with_classification"][beta]["conf"] = mst.confidences[best_thresh_idx]

            output_dict[epoch]["Total_statistics_with_classification"][beta]["fixed_f1"] = fix_best_f1
            output_dict[epoch]["Total_statistics_with_classification"][beta]["fixed_conf"] = mst.confidences[fix_best_thresh_idx]

            output_dict[epoch]["Total_statistics_with_classification"][beta]["mAP"] = mAP
            output_dict[epoch]["Total_statistics_with_classification"][beta]["fixed_mAP"] = fix_mAP

            output_dict[epoch]["Total_statistics_with_classification"][beta]["mean_iou"] = mean_iou
            output_dict[epoch]["Total_statistics_with_classification"][beta]["mean_hausdorff_distance"] = mean_hausdorff_distance

        # total statistic by size
        for size, size_name in enumerate(['small', 'medium', 'large']):
            mAP, best_f1, _, _, precision_list, recall_list = mst.get_map_and_best_scores(size=size, fixed=False, include_classification=True)
            fix_mAP, fix_best_f1, _, _, fix_precision_list, fix_recall_list = mst.get_map_and_best_scores(size=size, include_classification=True)

            if mAP != -1:
                create_empty_dict_key_if_required(output_dict[epoch], "Total_statistics_with_classification_per_size")
                create_empty_dict_key_if_required(output_dict[epoch]["Total_statistics_with_classification_per_size"], beta)

                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_precision_{size_name}"] = precision_list[best_pr_idx]
                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_recall_{size_name}"] = recall_list[best_pr_idx]
                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_f1_{size_name}"] = best_f1

                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_fixed_precision_{size_name}"] = fix_precision_list[fixed_best_pr_idx]
                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_fixed_recall_{size_name}"] = fix_recall_list[fixed_best_pr_idx]
                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"validation_fixed_f1_{size_name}"] = fix_best_f1

                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"mAP_{size_name}"] = mAP
                output_dict[epoch]["Total_statistics_with_classification_per_size"][beta][f"fixed_mAP_{size_name}"] = fix_mAP

    return output_dict