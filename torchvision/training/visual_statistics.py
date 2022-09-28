from multiprocessing import Process
from threading import Thread
from queue import Queue

import cv2

from building_statistics_utils import *
from runtime_utils import convert_buildings_data_to_mrcnn
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage.measure import find_contours
import time
import argparse
import numpy as np
import json

from building_models import models_registry
from utils import torch2numpy_image
from torchvision.models.detection.mask_rcnn import MaskRCNN
import run_model
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Queue to save the job parameters.
jobs_q = Queue()


def create_dataset_visualisation_work_creator(model, dataloader, path, device='cuda', classes=1, iou_thresh=0.5,
                                              scaling_factor=1):
    """
    create jobs by filling queue in jobs parameters, when finnish put None in the queue
    :param model:MaskRCNN, mask rcnn model
    :param dataloader:DataLoader, data loader object
    :param path:str, to save the images
    :param device:str, torch.device
    :param classes:int, number of classes
    :param scaling_factor: float, uses when analysis and inference made in different scales
    :param iou_thresh: float, iou threshold
    :return:
    """

    global jobs_q

    if not os.path.exists(path):
        os.mkdir(path)

    model.roi_heads.score_thresh = 0.001  # DOES NOT supports merged models!!!!
    model.eval()

    mst = MaskStatisticsTorch(iou_threshold=iou_thresh, dataset=dataloader.dataset, model=model, device=device,
                              classes=classes, scaling_factor=scaling_factor)

    # plot statistic results:
    mAP, best_f1, best_tresh_idx, best_pr_idx, precision_list, recall_list = mst.get_map_and_best_scores(fixed=False,
                                                                                                         include_classification=True)
    mAP_fixed, best_f1_fixed, best_tresh_idx_fixed, fixed_best_pr_idx, precision_list_fixed, recall_list_fixed = mst.get_map_and_best_scores(
        include_classification=True)

    print('Total results including classification:\n')
    print('fixed mAP={:.3f} \t mAP = {:.3f}'.format(mAP_fixed, mAP))
    print('fixed best f1={:.3f} \t best f1 = {:.3f}'.format(best_f1_fixed, best_f1))
    print('fixed precision={:.3f} \t precision = {:.3f}'.format(precision_list_fixed[fixed_best_pr_idx],
                                                                precision_list[best_pr_idx]))
    print(
        'fixed recall={:.3f} \t recall = {:.3f}'.format(recall_list_fixed[fixed_best_pr_idx], recall_list[best_pr_idx]))
    print('fixed confidence={:.3f} \t confidence = {:.3f}'.format(mst.confidences[best_tresh_idx_fixed],
                                                                  mst.confidences[best_tresh_idx]))

    print('\nTotal results detection only:\n')
    mAP_d, best_f1_d, best_tresh_idx_d, best_pr_idx_d, precision_list_d, recall_list_d = mst.get_map_and_best_scores(
        fixed=False, include_classification=False)

    mAP_fixed_d, best_f1_fixed_d, best_tresh_idx_fixed_d, fixed_best_pr_idx_d, precision_list_fixed_d, recall_list_fixed_d = mst.get_map_and_best_scores(
        include_classification=False)

    print('fixed mAP={:.3f} \t mAP = {:.3f}'.format(mAP_fixed_d, mAP_d))
    print('fixed best f1={:.3f} \t best f1 = {:.3f}'.format(best_f1_fixed_d, best_f1_d))
    print('fixed precision={:.3f} \t precision = {:.3f}'.format(precision_list_fixed_d[fixed_best_pr_idx_d],
                                                                precision_list_d[best_pr_idx_d]))
    print('fixed recall={:.3f} \t recall = {:.3f}'.format(recall_list_fixed_d[fixed_best_pr_idx_d],
                                                          recall_list[best_pr_idx_d]))
    print('fixed confidence={:.3f} \t confidence = {:.3f}'.format(mst.confidences[best_tresh_idx_fixed_d],
                                                                  mst.confidences[best_tresh_idx_d]))

    for size in range(3):
        print(f"\nsize {size} performence(include classification):")
        mAP, best_f1, _, _, precision_list, recall_list = mst.get_map_and_best_scores(size=size, fixed=False,
                                                                                      include_classification=True)
        mAP_fixed, best_f1_fixed, _, _, precision_list_fixed, recall_list_fixed = mst.get_map_and_best_scores(size=size,
                                                                                                              include_classification=True)
        print('\tfixed precision={:.3f} \t precision = {:.3f}'.format(precision_list_fixed[fixed_best_pr_idx],
                                                                      precision_list[best_pr_idx]))
        print('\tfixed recall={:.3f} \t recall = {:.3f}'.format(recall_list_fixed[fixed_best_pr_idx],
                                                                recall_list[best_pr_idx]))

        best_f1 = (1 + 1 ** 2) * (recall_list[best_pr_idx] * precision_list[best_pr_idx] / (
                1 ** 2 * precision_list[best_pr_idx] + recall_list[best_pr_idx] + 1e-6))
        best_f1_fixed = (1 + 1 ** 2) * (
                recall_list_fixed[fixed_best_pr_idx] * precision_list_fixed[fixed_best_pr_idx] / (
                1 ** 2 * precision_list_fixed[fixed_best_pr_idx] + recall_list_fixed[fixed_best_pr_idx] + 1e-6))

        print('\tfixed f1={:.3f} \t f1 = {:.3f}'.format(best_f1_fixed, best_f1))

        print(f"\nsize {size} performence(detection only):")
        mAP, best_f1, _, _, precision_list, recall_list = mst.get_map_and_best_scores(size=size, fixed=False,
                                                                                      include_classification=False)
        mAP_fixed, best_f1_fixed, _, _, precision_list_fixed, recall_list_fixed = mst.get_map_and_best_scores(size=size,
                                                                                                              include_classification=False)
        print('\tfixed precision={:.3f} \t precision = {:.3f}'.format(precision_list_fixed[fixed_best_pr_idx],
                                                                      precision_list[best_pr_idx]))
        print('\tfixed recall={:.3f} \t recall = {:.3f}'.format(recall_list_fixed[fixed_best_pr_idx],
                                                                recall_list[best_pr_idx]))

        best_f1 = (1 + 1 ** 2) * (recall_list[best_pr_idx] * precision_list[best_pr_idx] / (
                1 ** 2 * precision_list[best_pr_idx] + recall_list[best_pr_idx] + 1e-6))
        best_f1_fixed = (1 + 1 ** 2) * (
                recall_list_fixed[fixed_best_pr_idx] * precision_list_fixed[fixed_best_pr_idx] / (
                1 ** 2 * precision_list_fixed[fixed_best_pr_idx] + recall_list_fixed[fixed_best_pr_idx] + 1e-6))

        print('\tfixed f1={:.3f} \t f1 = {:.3f}'.format(best_f1_fixed, best_f1))

    mst.model = None
    mst.dataset = None

    print('start loading images...')

    for index, (data, image_name) in enumerate(dataloader):
        eval_data = convert_buildings_data_to_mrcnn(data, device, False)
        gt = data["targets"][0]
        gt_masks = gt['masks']

        preds = model(**eval_data)[0]
        pred_masks = preds['masks'].squeeze(1)

        h, w = data["main_input"][0].shape[1:]
        preds['masks'] = torch.where(preds['masks'] >= 0.5, torch.ones_like(preds['masks']),
                                     torch.zeros_like(preds['masks'])).resize(preds['masks'].size(0), h, w)

        # preparing the mask in the same size as in the mst object to avoid prediction mismatch
        if scaling_factor != 1:
            h = int(h * scaling_factor)
            w = int(w * scaling_factor)

            pred_masks = F.interpolate(pred_masks.float().unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
            gt_masks = F.interpolate(gt_masks.float().unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
            gt_masks = torch.where(gt_masks >= 0.5, torch.ones_like(gt_masks), torch.zeros_like(gt_masks))

        pred_masks = torch.where(pred_masks >= 0.5, torch.ones_like(pred_masks), torch.zeros_like(pred_masks))

        # filter the relevant prediction accordint to the correct scale
        _, relevant_pred = remove_masks_from_edges(pred_masks.permute(1, 2, 0).type(torch.cuda.IntTensor),
                                                   scaling_factor)
        _, relevant_gt = remove_masks_from_edges(gt_masks.permute(1, 2, 0).type(torch.cuda.IntTensor), scaling_factor)

        preds_np = {'masks': preds['masks'][relevant_pred == 1].cpu().numpy()}
        gt_np = {'masks': gt['masks'][relevant_gt == 1].cpu().numpy()}

        # put parameters in Queue for image visualization
        img = torch2numpy_image(eval_data["main_input"][0])
        jobs_q.put((cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX).astype(np.int32),
                    gt_np, preds_np, path, index, image_name, mst, best_tresh_idx_fixed))

        del eval_data["main_input"], preds, gt, relevant_pred, relevant_gt

    # None mark the end of the Queue
    jobs_q.put(None)

    print('all jobs set')


def create_image_visualization(img, gt, preds, path, index, image_name, mst, confidence_index, iou_thresh=0.5):
    """
    create image with prediction and gt visualization
    :param img:tensor, image
    :param gt:tensor, gt mask
    :param preds:np.array preds mask
    :param path:str, to save the images
    :param index:int, of the image
    :param image_name: str
    :param mst: MaskStatisticsTorch object
    :param confidence_index:int, for model best results
    :param iou_thresh:int, iou threshold
    :return:
    """

    print('start work on image {}'.format(index))
    contour_formating = {"tp": {"color": "green", "linestyle": "-"},
                         "tp_det": {"color": "green", "linestyle": "dashed"},
                         "contained": {"color": "blue", "linestyle": "-"},
                         "contained_det": {"color": "blue", "linestyle": "dashed"},
                         "contains": {"color": "orange", "linestyle": "-"},
                         "contains_det": {"color": "orange", "linestyle": "dashed"},
                         "fp": {"color": "red", "linestyle": "-"},
                         "fn": {"color": "red", "linestyle": "-"}
                         }

    TP = mlines.Line2D([], [], color='green', label='TP', linewidth=5)
    TP_DET = mlines.Line2D([], [], color='green', label='TP_DET', linestyle="--", linewidth=5)
    contained = mlines.Line2D([], [], color='blue', label='contained', linewidth=5)
    contained_det = mlines.Line2D([], [], color='blue', linestyle="--", label='contained_det', linewidth=5)
    containes = mlines.Line2D([], [], color='orange', label='contains', linewidth=5)
    FP = mlines.Line2D([], [], color='red', label='FP', linewidth=5)
    FN = mlines.Line2D([], [], color='red', label='FN', linewidth=5)

    pred_legend = [TP, TP_DET, FP, contained, containes]
    gt_legend = [TP, TP_DET, FN, contained, contained_det]

    result_array = mst.results_list[index][RL_RESULT_ARRAY]
    gt_pred_cover_list = mst.results_list[index][RL_GT_PRED_COVER]

    fig, axs = plt.subplots(1, 2, figsize=(40, 20))

    axs[0].imshow(np.transpose(img, (1, 2, 0)))
    axs[1].imshow(np.transpose(img, (1, 2, 0)))

    confidence = mst.confidences[confidence_index]

    # the next two if statements checks if a mismatch due to the remove mask from edges with different resolution (optional when scaling factor!=1)

    if preds['masks'].shape[0] != len(result_array):
        print(
            'image {} prediction mismatch: prediction number is {}, result array length is {} . image will not be save'.format(
                index, preds['masks'].shape[0], len(result_array)))
        exit()

    if gt['masks'].shape[0] != len(gt_pred_cover_list):
        print('image {} gt mismatch: gt number is {}, gt pred cover length is {} . image will not be save'.format(index,
                                                                                                                  gt[
                                                                                                                      'masks'].shape[
                                                                                                                      0],
                                                                                                                  len(gt_pred_cover_list)))
        exit()

    for i in range(preds['masks'].shape[0]):

        contour = find_contours(preds['masks'][i, :, :], level=0.5)

        if result_array[i][RA_CONFIDENCE] > confidence:
            if result_array[i][RA_TP_CLS_FLAG] == 1:  # the tp case
                for c in range(len(contour)):
                    axs[0].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["tp"])
            elif result_array[i][RA_TP_DET_FLAG] == 1:  # the tp case
                for c in range(len(contour)):
                    axs[0].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["tp_det"])

            elif result_array[i][RA_MULTIPLE_GTS] != 0:  # the prediction contains more than 1 GT
                for c in range(len(contour)):
                    axs[0].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["contains"])

            elif result_array[i][RA_GT_COMPONENT] != -1:  # the prediction is part of some GT together with more preds
                for c in range(len(contour)):
                    axs[0].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["contained"])

            else:  # fp
                for c in range(len(contour)):
                    axs[0].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["fp"])

    for i in range(gt['masks'].shape[0]):

        contour = find_contours(gt['masks'][i, :, :], level=0.5)  # level define by the mask thresh as 0.5

        gt_pred_cover = gt_pred_cover_list[i]
        # get the smallest confidence that passes the IOU thresh (or the gt contained in bigger prediction)
        j = len(gt_pred_cover) - 1

        if gt_pred_cover[0][GPC_PRED] != -2:
            best_conf_pred_index = gt_pred_cover[j][GPC_PRED]
            predicted_class = result_array[best_conf_pred_index, RA_CLASS]
            is_tp = False
            is_cov = False
            while not is_tp and j > -1:  # the tp_fix case
                if gt_pred_cover[j][GPC_COVERAGE_SUM] >= iou_thresh and gt_pred_cover[j][
                    GPC_PRED_CONFIDENCE] >= confidence:
                    is_tp = True
                    for c in range(len(contour)):
                        if predicted_class == gt_pred_cover[j][GPC_CLS]:
                            axs[1].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["tp"])
                        else:
                            axs[1].plot(contour[c][:, 1], contour[c][:, 0], linewidth=4, **contour_formating["tp_det"])
                elif gt_pred_cover[j][GPC_COVERAGE_SUM] == -1 and gt_pred_cover[j][GPC_PRED_CONFIDENCE] > confidence:
                    is_cov = True

                j -= 1

            if not is_tp and is_cov:  # the case where gt contain in bigger prediction:
                if predicted_class == gt_pred_cover[0][GPC_CLS]:
                    axs[1].plot(contour[0][:, 1], contour[0][:, 0], linewidth=4, **contour_formating["contained"])
                else:
                    axs[1].plot(contour[0][:, 1], contour[0][:, 0], linewidth=4, **contour_formating["contained_det"])

            elif not is_tp:  # the fn_fix case
                axs[1].plot(contour[0][:, 1], contour[0][:, 0], linewidth=4, **contour_formating["fn"])

        else:  # the fn_fix case
            axs[1].plot(contour[0][:, 1], contour[0][:, 0], linewidth=4, **contour_formating["fn"])

    tp_fix = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_TP_CLS].sum())
    fp_fix = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_FP_CLS].sum())
    gt_num = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_GT_NUM].sum())

    tp = int(mst.evaluation_tensor[index, confidence_index, :, ET_TP_CLS].sum())
    fp = int(mst.evaluation_tensor[index, confidence_index, :, ET_FP_CLS].sum())

    tp_fix_d = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_TP_DET].sum())
    fp_fix_d = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_FP_DET].sum())
    gt_num_d = int(mst.fixed_evaluation_tensor[index, confidence_index, :, ET_GT_NUM].sum())
    fn_fix_d = gt_num_d - tp_fix_d

    tp_d = int(mst.evaluation_tensor[index, confidence_index, :, ET_TP_DET].sum())
    fp_d = int(mst.evaluation_tensor[index, confidence_index, :, ET_FP_DET].sum())

    axs[0].set_title(
        'Prediction: cls: tp = {} ({}), fp = {} ({}) || det: tp = {} ({}), fp = {} ({})'.format(tp_fix, tp, fp_fix, fp,
                                                                                                tp_fix_d, tp_d,
                                                                                                fp_fix_d, fp_d),
        fontsize=30)
    axs[1].set_title('GT: gt_num = {}, fn = {}'.format(gt_num_d, fn_fix_d), fontsize=30)

    axs[0].legend(handles=pred_legend, bbox_to_anchor=(-0.2, 1), loc='upper left', prop={'size': 20})
    axs[1].legend(handles=gt_legend, bbox_to_anchor=(-0.2, 1), loc='upper left', prop={'size': 20})

    fig.savefig(path + '/{}_index_{}.png'.format(image_name, index), dpi=400)


def work_manager(process_number=5):
    """
    thread that create processes for parallel computetion
    :param process_number:int, number of processes to use
    :return:
    """
    global jobs_q
    proccess_list = []

    for i in range(process_number):
        args_i = jobs_q.get()
        proccess_list.append(Process(target=create_image_visualization, args=args_i))
        proccess_list[i].start()

    end = False

    while True:
        for i in range(process_number):
            if not proccess_list[i].is_alive():
                proccess_list[i].join()
                args_i = jobs_q.get()
                if args_i is None:
                    end = True
                    break
                proccess_list[i] = Process(target=create_image_visualization, args=args_i)
                proccess_list[i].start()
        if end:
            break
        time.sleep(2)


def run_visual_statistic(model, valid_loader, path, scaling_factor=1, classes=1):
    """
    function to create images with prediction and gt for visual results
    :param model:MaskRCNN, mask-rcnn model
    :param valid_loader: dataloader object
    :param path: path to save images
    :param scaling_factor: float scaling factor
    :return: saving the images in the given path
    """
    global jobs_q
    jobs_q = Queue()
    start = time.time()
    works = Thread(target=create_dataset_visualisation_work_creator,
                   args=(model, valid_loader, path, 'cuda', classes, 0.5, scaling_factor))
    works.start()

    manager = Thread(target=work_manager, args=(5,))
    manager.start()

    works.join()
    manager.join()

    print(f'total running time is {(time.time() - start) / 60} minutes')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_root", help='path to the root directory')
    parser.add_argument("--output_folder_name", default='outputs')
    parser.add_argument("--model_type", default="best_total",
                        help="choose which saved model to use, insert the dir name")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--scaling_factor", type=float, default=1)

    args = parser.parse_args()

    if not os.path.isdir(args.path_to_root):
        raise Exception("the given path doesn't lead to valid directory.")

    if args.model_type not in {"best_total", "best_small", "best_medium", "best_large", "last_epoch"}:
        raise Exception("model type is not legal.")

    json_config_file = args.path_to_root + "/checkpoints/run_config.json"
    model_checkpoint = args.path_to_root + "/checkpoints/" + args.model_type + "/maskrcnn_resnet_base"

    # Open and parse config file, note that we might desire to create a thin, designated config file for E2E which will be given by research team
    with open(json_config_file, "r") as in_json:
        config: dict = json.loads(in_json.read())

    # Build a torch.device object used to run all calculations on a cuda device
    device: torch.device = torch.device(args.device)

    # Build model from model registry
    # note that config['model'] is the name of the model which defines the function used in model registry
    model: MaskRCNN = models_registry[config['model']](config).to(device)

    # Open a given checkpoint file (pickle file) and load it's weights to the given model
    with open(model_checkpoint, "rb") as in_check:
        checkpoint: dict = torch.load(in_check)
        model.load_state_dict(checkpoint['model_state_dict'])

    # build datasets:
    train_ds, valid_ds = run_model.load_datasets(config)
    valid_loader = DataLoader(valid_ds, collate_fn=run_model.collate_fn)

    path = args.path_to_root + "/" + args.output_folder_name

    run_visual_statistic(model, valid_loader, path, args.scaling_factor)
