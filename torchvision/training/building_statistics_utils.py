import numpy as np
from caffe2.python.layers import cls
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import find_contours
import torch
import time
import torch.nn.functional as F

# constants used for column indexing of results table. see explanation for each in MaskStatistics docstring
RA_CONFIDENCE = 0
RA_CLASS = 1
RA_IOU = 2
RA_TP_DET_FLAG = 3
RA_TP_CLS_FLAG = 4
RA_GT_MATCHED = 5
RA_GT_COMPONENT = 6
RA_GT_COVERAGE_SUM = 7
RA_MULTIPLE_GTS = 8
RA_HAUSDORFF_DISTANCE_SCORE = 9
RA_AREA = 10
RESULT_TABLE_HEADERS = [RA_CONFIDENCE, RA_CLASS, RA_IOU, RA_TP_DET_FLAG, RA_TP_CLS_FLAG, RA_GT_MATCHED, RA_GT_COMPONENT,
                        RA_GT_COVERAGE_SUM, RA_MULTIPLE_GTS,
                        RA_HAUSDORFF_DISTANCE_SCORE, RA_AREA]

# constants used to choose relevant part from result list triplet (RL- Result List)
RL_RESULT_ARRAY = 0
RL_GT_PRED_COVER = 1
RL_NUM_GROUND = 2

# constants used to choose relevant part from gt_pred_cover list objects triplets (GPC- Gt Pred Cover)
GPC_PRED = 0
GPC_COVERAGE_SUM = 1
GPC_PRED_CONFIDENCE = 2
GPC_CLS = 3
GPC_AREA = 4

# constants used to choose from precision recall list (PRG- Percision Recall Graph)
PRG_RECALL = 0
PRG_PRECISION = 1
PRG_CONFIDENCE = 2
PRG_HEADERS = [PRG_RECALL, PRG_PRECISION, PRG_CONFIDENCE]

# buffer from edge of image to consider an object on the edge better to set as a power of 2 to minimize possible mismatch
IMAGE_EDGE_BUFFER = 8
# minimum IoU to consider an object to be part of a group of objects comprising a larger object
MIN_IOU_FOR_MATCHING = 0.05
# when checking if an object is fully contained by another, we allow a small difference between the intersection and the original object
OBJECT_CONTAIN_DIFF_THRESH = 0.3
# when checking if an object is small and on an edge of an image, small is defined as a factor of pixels from the entire image
# 0.0002 represent 16**2 pixcells object in 1024x1024 image.
FACTOR_FROM_IMAGE = 0.0002

# hausdorff distance normalization factor.
# by normalizing by this factor, the distance is calculated by max(1 - num_of_pixels_highest_distance/100, 0)
# for example, if HD score is 0.87 then the farthest distance is 13 pixels.
HD_DIST_NORMALIZE_CLIPPING = 100

# key for evaluation tensor:
ET_TP_DET = 0
ET_FP_DET = 1
ET_TP_CLS = 2
ET_FP_CLS = 3
ET_GT_NUM = 4

# sizes keys for evaluation tensor
SMALL = 0
MEDIUM = 1
LARGE = 2


def pad_mask(mask):
    """
    Pads a 2d numpy array with zeros
    :param mask: 2d numpy array of a given mask
    :retrun: padded mask with zeros
    """
    return np.pad(mask, 1, 'constant', constant_values=0)


def calculate_hausdorff_distance(pred_mask, gt_mask):
    """
    Returns the hasudorff distance between the contours of 2 masks.
    Note that if there is more than 1 contour for one of the masks, the longest contour is chosen.
    :param pred_mask: ndarray size of (W,H) with prediction mask
    :param gt_mask: ndarray size of (W,H) with gt mask
    :return: hasudorff distance between contours
    """
    gt_contours = find_contours(gt_mask, 0)
    gt_contour = gt_contours[0] if len(gt_contours) == 1 else gt_contours[
        np.argmax([x.shape[0] for i, x in enumerate(gt_contours)])]

    pred_contours = find_contours(pred_mask, 0)
    pred_contour = pred_contours[0] if len(pred_contours) == 1 else pred_contours[
        np.argmax([x.shape[0] for i, x in enumerate(pred_contours)])]

    return max(directed_hausdorff(gt_contour, pred_contour)[0], directed_hausdorff(pred_contour, gt_contour)[0])


def get_intersection_to_larger_mask(mask, designated_mask, contained_condition=False):
    """
    This function retruns the intersection between a mask and a designated mask.
    If contained condition is True, the function checks that the mask is contained within the designated_mask by less then OBJECT_CONTAIN_DIFF_THRESH
    and will return a zeros matrix the size of the mask.
    :param mask: tensor of size (W,H)
    :param designated_mask: tensor of size (W,H)
    :param contained_condition: boolean signaling if to check that the mask is contained within the designated mask
    :return: the intersection between the mask and the designated mask.
             In the case where contained condition is true and the condition is not fullfilled will return a zeros tensor.
    """
    intersection = mask & designated_mask
    if contained_condition:
        object_contain_diff = mask.sum((0, 1)).float() / intersection.sum((0, 1)).float()
        if np.abs(1 - object_contain_diff.cpu().numpy()) > OBJECT_CONTAIN_DIFF_THRESH:
            return torch.zeros_like(mask).int()

    return intersection


def remove_masks_from_edges(mask, edge_factor=1):
    """
    This function removes masks which are small and on the edge of an image.
    Small is defined by a factor from the image- FACTOR_FROM_IMAGE
    The edge is defined by up to IMAGE_EDGE_BUFFER pixels from the edge
    :param mask: tensor of shape (W, H, N) with N masks
    :param edge_factor: float increase/decrese the basic edge buffer, in use when apply statistic on different resolution then the inference
    :return: a tuple with the tensor of size (W, H, M) where M is the number of relevant masks after removals and
             a tensor with numbers of relevant masks
    """
    # mask to find which masks are on edges
    image_edge_mask = torch.ones_like(mask)

    scailed_buffer = int(IMAGE_EDGE_BUFFER * edge_factor)

    image_edge_mask[scailed_buffer:-scailed_buffer, scailed_buffer:-scailed_buffer, :] = 0

    # intersect mask with image_mask_edge and get all masks which are on edge
    masks_intersected = mask & image_edge_mask
    irrelevant_masks_proposed_edge = torch.sum(masks_intersected, axis=(0, 1)) != 0

    # check for small masks
    irrelevant_masks_proposed_small = torch.sum(mask, axis=(0, 1)) < (
                FACTOR_FROM_IMAGE * image_edge_mask.shape[0] * image_edge_mask.shape[1])

    # check for all small masks on edge and remove them
    irrelevant_masks = irrelevant_masks_proposed_edge & irrelevant_masks_proposed_small
    relevant_masks = irrelevant_masks != 1
    mask = mask[:, :, relevant_masks]

    return mask, relevant_masks


class MaskStatistics:
    """
    Class to calculate statistics on a given dataset with a given model.
    Calculates MAP (Mean Average Precision), housdorf distance, mean IOU, and f1 score.
    possible to filter by size or class
    fixed results relate to the merge
    """

    def __init__(self, iou_threshold, image_ids, path=None, calculate_hausdorff=True, classes=1,
                 size_thresh=(32 ** 2, 96 ** 2), scaling_factor=1):
        """
        :param iou_threshold: float, IOU threshold for defining a TP
        :param image_ids: int with number of images in dataset
        :param path: string path. if given ist the the path for saving a recall precision graph.
        :param calculate_hausdorff: boolean, define if calculate housdorff in the statistic
        :param classes: int, number of classes in the data set (not including background)
        :param size_thresh: tuple, define the range of sizes
        :param scaling_factor: float, scaling factor for analysis with different resolution
        """

        self.iou_threshold = iou_threshold
        self.image_ids = image_ids
        self.calculate_hausdorff = calculate_hausdorff
        self.path = path
        self.classes = classes
        self.size_tresh = size_thresh
        self.scaling_factor = scaling_factor

        if len(size_thresh) != 2:
            raise Exception("area thresh need to be exactly in length 2 (3 available sizes)")
        # result_list[img_mun][results_array/gt_pred_cover]
        # results_array - information about the predictions
        # gt_pred_cover - information about GTs
        # more details and explanation about statistics logic in the building_statistic_explained.docx file.
        self.results_list = []

        # confidence scores to calc mAP
        self.confidences = np.arange(0, 1.01, 0.01)
        self.confidences[100] = 0.999

        # numpy array to store evaluation results of shape: [img, conf, 3 for sizes (small/medium/large), 5 represent TP_DET/TP_CLS/FP_DET/FP_CLS/GT_NUM]
        self.evaluation_tensor = np.zeros((len(self.image_ids), len(self.confidences), 3, 5))

        # same as evaluation tensor, this time for fixed results (including merging and spliting results)
        self.fixed_evaluation_tensor = np.zeros((len(self.image_ids), len(self.confidences), 3, 5))

        # fill results list
        self.get_statistics_results_for_ds()

        # fill evaluation tensors:
        self.evaluate()
        self.evaluate_fixed_results()

    def get_fixed_confusion_matrix(self, thresh):
        """
        :param thresh: float, score threshold of the model
        :return: calculate the fixed confusion matrix
        """

        # reset confusion matrix
        fixed_confusion_matrix = np.zeros((self.classes + 1, self.classes + 1)).astype(int)

        # fill the FP part when fail in detection (predicting background instead class)
        for img_idx, result in enumerate(self.results_list):
            for prediction in range(len(result[RL_RESULT_ARRAY])):
                if result[RL_RESULT_ARRAY][prediction, RA_CONFIDENCE] >= thresh:
                    # if not contain in bigger prediction and not contains multiple gts
                    if result[RL_RESULT_ARRAY][prediction, RA_GT_COMPONENT] == -1 and result[RL_RESULT_ARRAY][
                        prediction, RA_MULTIPLE_GTS] == 0:
                        fixed_confusion_matrix[int(result[RL_RESULT_ARRAY][prediction, RA_CLASS]), 0] += 1

        # fill the TPs and FP_CLS (FP when detection is correct but wrong classification has been made) parts from the GT PRED COVER table
        for img_idx, result in enumerate(self.results_list):
            for gt_pred_cover in (result[RL_GT_PRED_COVER]):

                # get the highest confidence (last line)
                i = len(gt_pred_cover) - 1

                # get the highest confidence predicted class: (if no prediction exists (GPC_PRED=-2), skip this section and add as fn to the matrix)
                if gt_pred_cover[i][GPC_PRED] != -2:
                    best_conf_pred_index = gt_pred_cover[i][GPC_PRED]
                    predicted_class = self.results_list[img_idx][RL_RESULT_ARRAY][best_conf_pred_index, RA_CLASS]

                    is_tp = False
                    while is_tp is False and i > -1:

                        # the cases where there is a detection to this gt
                        if gt_pred_cover[i][GPC_COVERAGE_SUM] >= self.iou_threshold or gt_pred_cover[i][
                            GPC_COVERAGE_SUM] == -1:
                            is_tp = True

                            if gt_pred_cover[i][GPC_PRED_CONFIDENCE] > thresh:
                                fixed_confusion_matrix[int(predicted_class), int(gt_pred_cover[i][GPC_CLS])] += 1
                            else:
                                fixed_confusion_matrix[0, int(gt_pred_cover[i][GPC_CLS])] += 1

                        i -= 1

                    #  if no valid prediction exists, set as fn in the matrix
                    if not is_tp:
                        fixed_confusion_matrix[0, int(gt_pred_cover[0][GPC_CLS])] += 1

                else:
                    fixed_confusion_matrix[0, int(gt_pred_cover[0][GPC_CLS])] += 1

        return fixed_confusion_matrix

    def get_confusion_matrix(self, thresh):
        """
        :param thresh: float, score threshold of the model
        :return: calculate the fixed confusion matrix
        """

        # reset confusion matrix
        confusion_matrix = np.zeros((self.classes + 1, self.classes + 1)).astype(int)

        for img_idx, result in enumerate(self.results_list):

            gt_array = np.zeros(len(result[
                                        RL_GT_PRED_COVER]))  # array to store witch gt has been detected (the zeros value at the end will be the FN)
            for prediction in range(len(result[RL_RESULT_ARRAY])):
                if result[RL_RESULT_ARRAY][prediction, RA_CONFIDENCE] > thresh:
                    if result[RL_RESULT_ARRAY][
                        prediction, RA_TP_DET_FLAG] == 1:  # the case where the detection is correct
                        gt_idx = int(result[RL_RESULT_ARRAY][prediction, RA_GT_MATCHED])
                        confusion_matrix[int(result[RL_RESULT_ARRAY][prediction, RA_CLASS]), int(
                            result[RL_GT_PRED_COVER][gt_idx][0][GPC_CLS])] += 1
                        gt_array[gt_idx] += 1
                    else:  # the case where it fails in detection (classify as gt = "background")
                        confusion_matrix[int(result[RL_RESULT_ARRAY][prediction, RA_CLASS]), 0] += 1

            fn_idx = np.argwhere(gt_array == 0)  # get the FN indexes (the zeros in the gt_array)
            if len(fn_idx) > 0:
                for i in range(len(fn_idx)):
                    idx = fn_idx[i][0]
                    confusion_matrix[
                        0, int(result[RL_GT_PRED_COVER][idx][0][GPC_CLS])] += 1  # classify as prediction = "background"

        return confusion_matrix

    def get_gt_num(self, size=-1, fixed=True):
        """
        calculate num of GT from the table
        :param size: int, size to count (only 1), if -1 count all sizes
        :param fixed: bool, define if we calculate fixed or regular statistics
        :return: num of GT
        """

        # choose the requested tensor
        evaluation_tensor = self.fixed_evaluation_tensor if fixed else self.evaluation_tensor

        # check for each case
        if size == -1:
            return evaluation_tensor[:, 0, :, ET_GT_NUM].sum()
        else:
            return evaluation_tensor[:, 0, size, ET_GT_NUM].sum()

    def get_tp_num(self, thresh_idx, size=-1, fixed=True, include_classification=True):
        """
        calculate num of GT from the table
        :param thresh_idx: int, the idx of the require tresh
        :param size: int, size to count (only 1), if -1 count all sizes
        :param fixed: bool, define if we calculate fixed or regular statistics
        :param include_classification: bool, determine if the tp including classification or detection only
        :return: int, num of TP
        """

        # choose the requested tensor
        evaluation_tensor = self.fixed_evaluation_tensor if fixed else self.evaluation_tensor

        tp = ET_TP_CLS if include_classification else ET_TP_DET

        # check for each case
        if size == -1:
            return evaluation_tensor[:, thresh_idx, :, tp].sum()

        else:
            return evaluation_tensor[:, thresh_idx, size, tp].sum()

    def get_fp_num(self, thresh_idx, size=-1, fixed=True, include_classification=True):
        """
        calculate num of GT from the table
        :param thresh_idx: int, the idx of the require tresh
        :param size: int, size to count (only 1), if -1 count all sizes
        :param fixed: bool, define if we calculate fixed or regular statistics
        :param include_classification: bool, determine if the fp including classification or detection only
        :return: int, num of FP
        """
        # choose the requested tensor
        evaluation_tensor = self.fixed_evaluation_tensor if fixed else self.evaluation_tensor

        fp = ET_FP_CLS if include_classification else ET_FP_DET

        if size == -1:
            return evaluation_tensor[:, thresh_idx, :, fp].sum()

        else:
            return evaluation_tensor[:, thresh_idx, size, fp].sum()

    def get_map_and_best_scores(self, size=-1, beta=1, fixed=True, include_classification=True):
        """
        :param size: int, size to count (only 1), if -1 count all sizes
        :param beta: float, to weight the f1 score
        :param fixed: bool, define if we calculate fixed or regular statistics
        :param include_classification: bool, determine if the statistics including classification or detection only
        :return: mAP, best f1 score, index and tresh and adjusted precision and recall
        """
        gt = self.get_gt_num(size=size)

        if gt == 0:
            print(f'no gt matches for cls = {cls} and size = {size}')
            return -1, -1, -1, -1, -1, -1

        precision_list = []
        recall_list = []
        best_f1 = 0
        best_tresh_idx = 0
        best_pr_idx = 0
        mAP = 0

        for idx, thresh_idx in enumerate(reversed(range(len(self.confidences)))):

            tp = self.get_tp_num(thresh_idx, size=size, fixed=fixed, include_classification=include_classification)
            fp = self.get_fp_num(thresh_idx, size=size, fixed=fixed, include_classification=include_classification)
            precision_list.append(tp / (tp + fp + 1e-6))
            recall_list.append(tp / (gt + 1e-6))

            f1 = (1 + beta ** 2) * (recall_list[idx] * precision_list[idx] / (
                        beta ** 2 * precision_list[idx] + recall_list[idx] + 1e-6))
            if f1 > best_f1:
                best_f1 = f1
                best_tresh_idx = thresh_idx
                best_pr_idx = idx

        for idx in range(len(self.confidences)):
            # 0 for the first point, current point otherwise
            left_x = 0 if idx == 0 else recall_list[idx]

            # 1 for the last point, next point otherwise
            right_x = recall_list[idx + 1] if idx + 1 < len(self.confidences) else 1

            mAP += (right_x - left_x) * precision_list[idx]

        return mAP, best_f1, best_tresh_idx, best_pr_idx, precision_list, recall_list

    def get_sizes(self, area):
        """
        :param area: np array, areas of the instances
        :return: np array, classification as SMALL/MEDIUM/LARGE
        """
        sizes = np.zeros_like(area)
        sizes += MEDIUM
        sizes[area < self.size_tresh[0]] = SMALL
        sizes[area > self.size_tresh[1]] = LARGE

        return sizes

    @staticmethod
    def get_conf_idx(conf):
        """
        :param conf: float, confidence
        :return: int, index of the confidence
        """
        if conf > 0.999:
            return 100

        else:
            return int(100 * conf)

    def evaluate(self):
        """
        :return: fill evaluation tensor tp and fp from results array
        """
        for img_idx, result in enumerate(self.results_list):
            for prediction in range(len(result[RL_RESULT_ARRAY])):
                # get index to fill the cells up to this index (all thresholds values below this index will consider this prediction as valid prediction)
                conf_idx = self.get_conf_idx(result[RL_RESULT_ARRAY][prediction, RA_CONFIDENCE])

                if result[RL_RESULT_ARRAY][prediction, RA_TP_DET_FLAG] == 1:  # the case where the detection is correct
                    self.evaluation_tensor[img_idx, 0:(conf_idx + 1), int(result[RL_RESULT_ARRAY][prediction, RA_AREA]),
                    ET_TP_DET] += 1
                    if result[RL_RESULT_ARRAY][
                        prediction, RA_TP_CLS_FLAG] == 1:  # the case where the detection and class correct
                        self.evaluation_tensor[img_idx, 0:(conf_idx + 1),
                        int(result[RL_RESULT_ARRAY][prediction, RA_AREA]), ET_TP_CLS] += 1
                    else:  # the case where detection is correct but class is wrong
                        self.evaluation_tensor[img_idx, 0:(conf_idx + 1),
                        int(result[RL_RESULT_ARRAY][prediction, RA_AREA]), ET_FP_CLS] += 1

                else:  # the case where it fails in detection
                    self.evaluation_tensor[img_idx, 0:(conf_idx + 1), int(result[RL_RESULT_ARRAY][prediction, RA_AREA]),
                    ET_FP_DET] += 1
                    self.evaluation_tensor[img_idx, 0:(conf_idx + 1), int(result[RL_RESULT_ARRAY][prediction, RA_AREA]),
                    ET_FP_CLS] += 1

    def evaluate_fixed_results(self):
        """
        :return: fill fixed evaluation tensor: tp form GT pred cover and fp from results array
        """
        self.fixed_evaluation_tensor[:, :, :, ET_GT_NUM] = self.evaluation_tensor[:, :, :, ET_GT_NUM]
        # fill the FP part when fail in detection
        for img_idx, result in enumerate(self.results_list):
            for prediction in range(len(result[RL_RESULT_ARRAY])):
                conf_idx = self.get_conf_idx(result[RL_RESULT_ARRAY][prediction, RA_CONFIDENCE])
                # if not = -1 then ist a TP or it related to some GT by merge/split operation then we count as TP_DET as well
                if result[RL_RESULT_ARRAY][prediction, RA_GT_COMPONENT] == -1 and result[RL_RESULT_ARRAY][
                    prediction, RA_MULTIPLE_GTS] == 0:
                    self.fixed_evaluation_tensor[img_idx, 0:(conf_idx + 1),
                    int(result[RL_RESULT_ARRAY][prediction, RA_AREA]), ET_FP_DET] += 1
                    self.fixed_evaluation_tensor[img_idx, 0:(conf_idx + 1),
                    int(result[RL_RESULT_ARRAY][prediction, RA_AREA]), ET_FP_CLS] += 1

        # fill the TPs and FP_CLS (FP when detection is correct but wrong classification has been made) parts from the GT PRED COVER table
        for img_idx, result in enumerate(self.results_list):
            for gt_pred_cover in (result[RL_GT_PRED_COVER]):

                # get the highest confidence prediction (last line in the table)
                i = len(gt_pred_cover) - 1

                # if its == -2 than no relevant prediction is exists
                if gt_pred_cover[i][GPC_PRED] != -2:
                    best_conf_pred_index = gt_pred_cover[i][GPC_PRED]
                    predicted_class = self.results_list[img_idx][RL_RESULT_ARRAY][best_conf_pred_index, RA_CLASS]

                    is_tp = False
                    while is_tp is False and i > -1:

                        # the cases where there is a detection to this gt
                        if gt_pred_cover[i][GPC_COVERAGE_SUM] >= self.iou_threshold or gt_pred_cover[i][
                            GPC_COVERAGE_SUM] == -1:
                            is_tp = True
                            conf_idx = self.get_conf_idx(gt_pred_cover[i][GPC_PRED_CONFIDENCE])

                            area = int(gt_pred_cover[i][GPC_AREA])

                            self.fixed_evaluation_tensor[img_idx, 0:(conf_idx + 1), area, ET_TP_DET] += 1

                            if gt_pred_cover[i][GPC_CLS] == predicted_class:
                                self.fixed_evaluation_tensor[img_idx, 0:(conf_idx + 1), area, ET_TP_CLS] += 1
                            else:
                                self.fixed_evaluation_tensor[img_idx, 0:(conf_idx + 1), area, ET_FP_CLS] += 1

                        i -= 1

    def get_mean_iou(self, thresh):
        """
        Gets mean IoU for a certain threshold
        Note that the IoU is calculated over predictions which are considered TP.
        :param thresh: confidence threshold to be used
        :return: mean IoU
        """
        stacked_result_list = []

        # for each result table (per image) get the predictions with confidence above thresh
        for result in self.results_list:
            # find position in current result table up to where confidence thresh is given and append table to stacked results list
            if result[RL_RESULT_ARRAY].shape[0] != 0:
                idx = np.argmin(np.abs(result[RL_RESULT_ARRAY][:, RA_CONFIDENCE] - thresh))
                idx = idx + 1 if result[RL_RESULT_ARRAY][idx, RA_CONFIDENCE] >= thresh else idx
                stacked_result_list.append(result[RL_RESULT_ARRAY][:idx])

        # stack all tables into 1 and sort from highest to lowest
        stacked_result_list = np.vstack(stacked_result_list)
        stacked_result_list = np.flipud(np.array(sorted(stacked_result_list, key=lambda x: x[0])))

        # calculate mean iou up to that position, if no tp's exist, mean_iou is set to 0
        tp_slices = stacked_result_list[np.where(stacked_result_list[:, RA_TP_DET_FLAG]), RA_IOU]
        if tp_slices.size > 0:
            mean_iou = np.mean(tp_slices)
        else:
            mean_iou = 0

        return mean_iou

    def get_mean_hausdorff_distance(self, thresh):
        """
        Gets mean HD distance for a certain threshold
        Note that the HD distance is calculated over all GT objects which are considered TP
        :param thresh: confidence threshold to be used
        :return: mean hausdorff distance
        """
        stacked_result_list = []

        # for each result table (per image) get the predictions with confidence above thresh
        for result in self.results_list:
            # find position in current result table up to where confidence thresh is given and append table to stacked results list
            if result[RL_RESULT_ARRAY].shape[0] != 0:
                idx = np.argmin(np.abs(result[RL_RESULT_ARRAY][:, RA_CONFIDENCE] - thresh))
                idx = idx + 1 if result[RL_RESULT_ARRAY][idx, RA_CONFIDENCE] >= thresh else idx
                stacked_result_list.append(result[RL_RESULT_ARRAY][:idx])

        # stack all tables into 1 and sort from highest to lowest
        stacked_result_list = np.vstack(stacked_result_list)
        stacked_result_list = np.flipud(np.array(sorted(stacked_result_list, key=lambda x: x[0])))

        # calculate mean hd_distance up to that position, if no slices exist, set distance to max which is HD_DIST_NORMALIZE_CLIPPING
        tp_slices = stacked_result_list[np.where(stacked_result_list[:, RA_TP_DET_FLAG]), RA_HAUSDORFF_DISTANCE_SCORE]
        if tp_slices.size > 0:
            hd_distance = np.mean(tp_slices)
        else:
            hd_distance = HD_DIST_NORMALIZE_CLIPPING

        return hd_distance

    def get_statistics_results_for_ds(self):
        """
        Fills the attribute of the results list for all images in dataset
        :return: None
        """
        # Iterate over all images in images_ids and get result table.
        images_it = tqdm(self.image_ids, position=0)
        images_it.set_description("Statistics Calculation")

        t_1 = 0
        t_2 = 0
        t_3 = 0
        t_4 = 0
        t_5 = 0

        for image_index in images_it:
            results, d5, d4, d3, d2, d1 = self.get_results_table_for_image(image_index)

            t_1 += d1
            t_2 += d2
            t_3 += d3
            t_4 += d4
            t_5 += d5

            self.results_list.append(results[0])

        # TODO: after final decision on multi process implementation, all time counters need to be remove

        print(f't1 = {t_1}')
        print(f't2 = {t_2}')
        print(f't3 = {t_3}')
        print(f't4 = {t_4}')
        print(f't5 = {t_5}')

    def fill_result_array(self, iou_matrix, gt_index, pred_index, p_scores, p_mask_t, p_sizes, p_cls, gt_mask_t,
                          gt_sizes, gt_cls):
        """
        Function gets gt and prediction's masks scores and IoU matrix, and returns a set of a result array and a gt prediction cover list.
        See docstring of MaskStatistics for further explanation.
        :param iou_matrix: tensor of size (num_predictions, num_ground) with Iou between all GT's and predictions
        :param gt_index: tensor of gt indices which match to predictions in pred_index
        :param pred_index: tensor of pred indices which match to predictions in gt_index
        :param p_scores: tensor of size (num_predictions) with score (confidence) per prediction
        :param gt_mask_t: tensor of size (W, H, num_ground) with gt masks
        :param p_sizes, int, pred sizes S/M/L
        :param p_cls, int, predictions classes
        :param p_mask_t: tensor of size (W, H, num_preds) with prediction masks
        :param gt_sizes: int, gt sizes S/M/L
        :return: objects of result_array, gt_pred_cover
        """
        num_pred = p_mask_t.shape[2]
        num_ground = gt_mask_t.shape[2]
        image_shape = gt_mask_t[:, :, 0].shape

        # initialize result array
        result_array = np.zeros((num_pred, len(RESULT_TABLE_HEADERS)))

        # set scores and classes of the predictions and IoU with corresponding gts
        result_array[:, RA_CONFIDENCE] = p_scores
        result_array[:, RA_CLASS] = p_cls
        result_array[pred_index, RA_IOU] = iou_matrix[gt_index, pred_index]

        # change TP_DET_FLAG = 1 for predictions with IOU above threshold
        result_array[:, RA_TP_DET_FLAG] = result_array[:, RA_IOU] > self.iou_threshold

        # add matching GT's per prediction
        result_array[pred_index, RA_GT_MATCHED] = gt_index[:]

        # change TP_CLS_FLAG = 1 for predictions with IOU above threshold and correct class prediction

        tp_cls_flags_val = (
                    gt_cls[result_array[:, RA_GT_MATCHED][result_array[:, RA_TP_DET_FLAG] == 1].astype(np.int)] ==
                    result_array[:, RA_CLASS][result_array[:, RA_TP_DET_FLAG] == 1]).astype(np.int64)

        result_array[:, RA_TP_CLS_FLAG][result_array[:, RA_TP_DET_FLAG] == 1] = tp_cls_flags_val
        # set sizes value by prediction size, if tp, value will determine by gt
        result_array[:, RA_AREA] = p_sizes

        result_array[:, RA_AREA][result_array[:, RA_TP_DET_FLAG] == 1] = gt_sizes[
            result_array[:, RA_GT_MATCHED][result_array[:, RA_TP_DET_FLAG] == 1].astype(np.int)]

        # get prediction confidence order
        pred_conf_order = np.argsort(result_array[:, RA_CONFIDENCE])[::-1]

        # change matching GT to -1 for predictions which aren't really matched
        # note that 0 is also a GT to be matched to
        result_array[(result_array[:, RA_TP_DET_FLAG] == 0), RA_GT_MATCHED] = -1
        result_array[:, RA_GT_COMPONENT] = -1

        # matrix holding gt's cover and list of indices covering them
        gt_coverage = torch.zeros((num_ground, image_shape[0], image_shape[1])).int().cuda()
        gt_pred_cover = [[] for _ in range(num_ground)]

        # iterate over every prediction from the highest to the lowest
        for pred in pred_conf_order:

            # get all gt's which intersect with this prediction
            relevant_gts = np.argwhere(iou_matrix[:, pred] > 0)

            # object to hold summation of all gt's covering this prediction
            combined_gt_objects = torch.zeros(image_shape).int().cuda()

            # in case a prediction is covered by several gt's we save them so we can later add them to the gt_pred_cover list
            add_to_solution = np.empty(0, dtype=np.int)

            # iterate over the relevant gt's
            for poss_gt in relevant_gts[0]:

                # get the intersection of the current prediction with the GT, for TP's we don't require the prediction to be contained in GT
                contained_condition = False if result_array[pred, RA_GT_MATCHED] == poss_gt else True
                intersection_with_gt = get_intersection_to_larger_mask(p_mask_t[:, :, pred],
                                                                       gt_mask_t[:, :, poss_gt],
                                                                       contained_condition=contained_condition)

                # if intersection exists, we add it to the coverage of this GT and the prediction save it in the gt_pred_cover
                if intersection_with_gt.sum((0, 1)) != 0:
                    gt_coverage[poss_gt] |= intersection_with_gt
                    result_array[pred, RA_GT_COVERAGE_SUM] = gt_coverage[poss_gt].sum((0, 1)).float() / gt_mask_t[:, :,
                                                                                                        poss_gt].sum(
                        (0, 1)).float()
                    gt_pred_cover[poss_gt].insert(0, [pred, result_array[pred, RA_GT_COVERAGE_SUM],
                                                      result_array[pred, RA_CONFIDENCE], gt_cls[poss_gt],
                                                      gt_sizes[poss_gt]])
                    result_array[pred, RA_GT_COMPONENT] = poss_gt

                # check if the GT is contained in the prediction and if so we add it to a combined GT mask
                intersection_with_pred = get_intersection_to_larger_mask(gt_mask_t[:, :, poss_gt],
                                                                         p_mask_t[:, :, pred],
                                                                         contained_condition=True)
                if intersection_with_pred.sum((0, 1)) != 0:
                    combined_gt_objects |= intersection_with_pred
                    add_to_solution = np.append(add_to_solution, poss_gt.numpy().astype(np.int))

            # check the intersection between the combined_gt_object and the prediction, if it is above iou_threshold count this as multiple GT's
            intersection = (combined_gt_objects & p_mask_t[:, :, pred]).sum().float()
            union = (combined_gt_objects | p_mask_t[:, :, pred]).sum().float()
            iou = intersection / union
            if iou > self.iou_threshold:

                # if above thresh add prediction to all gt's which are not a matched prediction in gt_pred_cover, -1 means this gt is contained in a bigger prediction
                for gt_for_solution in add_to_solution:
                    if result_array[pred, RA_GT_MATCHED] != gt_for_solution:
                        gt_pred_cover[gt_for_solution].insert(0, [pred, -1, result_array[pred, RA_CONFIDENCE],
                                                                  gt_cls[gt_for_solution], gt_sizes[gt_for_solution]])

                        # add to prediction how many gt's are matched
                        # note the exact number is not used for anything so this could have even been set as just a positive number
                        result_array[pred, RA_MULTIPLE_GTS] += 1

        # this is the case where no maching is found for the gt, we set -2 because -1 already in use for the contained in bigger pred
        for idx, gt_pred in enumerate(gt_pred_cover):
            if gt_pred == []:
                gt_pred.insert(0, [-2, -2, -2, gt_cls[idx], gt_sizes[idx]])

        return result_array, gt_pred_cover

    def update_gt_statistic_in_evaluation_tensor(self, index, gt_sizes):
        """
        :param index: int, index of the image
        :return:
        """
        for size in gt_sizes:
            self.evaluation_tensor[index, :, size, ET_GT_NUM] += 1

    def get_results_table_for_image(self, index):
        """
        for a given index out of a dataset, returns object of result_array, gt_pred_cover, num_ground (see docstring of MaskStatistics for futher details)
        :param index: index of a given image in a dataset
        :return:
        """

        start = time.time()
        # get ground truth
        gt_class, gt_area, gt_mask, image, depth = self.get_image_with_gt(index)

        # get predictions for image
        p_class, p_mask_t, p_scores = self.get_prediction(image, depth)

        t1 = time.time()

        # move predicted and GT masks to GPU for fast iou_matrix calculation
        gt_mask_t = gt_mask.to(torch.device('cuda')).type(torch.cuda.IntTensor)

        # filter out irrelevant gt's and predictions
        p_mask_t, relevant_preds = remove_masks_from_edges(p_mask_t, self.scaling_factor)
        relevant_preds = relevant_preds.cpu().numpy()

        p_scores = p_scores[relevant_preds == 1]
        p_class = p_class[relevant_preds == 1]

        p_area = p_mask_t.sum(axis=(0, 1)).cpu().numpy() * (1 / self.scaling_factor ** 2)
        p_sizes = self.get_sizes(p_area)

        gt_mask_t, relevant_gts = remove_masks_from_edges(gt_mask_t, self.scaling_factor)
        relevant_gts = relevant_gts.cpu().numpy()

        gt_class = gt_class[relevant_gts == 1].cpu().numpy()
        gt_area = gt_area[relevant_gts == 1].cpu().numpy()
        gt_sizes = self.get_sizes(gt_area)

        t2 = time.time()

        result_list = []

        # update gt statistic in the evaluation tensor - not related to the result table
        self.update_gt_statistic_in_evaluation_tensor(index, gt_sizes)

        # the case where no ground true exists (empty image). gt_pred will be empty, all predictions will be fp
        if gt_class.shape[0] == 0:
            result_array = np.zeros((p_scores.shape[0], len(RESULT_TABLE_HEADERS)))
            result_array[:, RA_CONFIDENCE] = p_scores
            result_array[:, RA_GT_COMPONENT] = -1
            result_array[:, RA_AREA] = p_sizes
            result_array[:, RA_CLASS] = p_class

            result_list.append((result_array, [], 0))
            t3 = time.time()
            t4 = time.time()
            t5 = time.time()

            return result_list, t5 - t4, t4 - t3, t3 - t2, t2 - t1, t1 - start

        # calculate intersection and union vectors for the cls for each gt_mask and then stack them together
        # note that this running this in torch on GPU while iterating for each gt mask has better performance
        # than broadcasting all GT and predicted masks into 1 large matrix and calculating them all at once
        iou_vectors = []
        inter_vectors = []
        for mid in range(gt_mask_t.shape[2]):
            inter_vec = (gt_mask_t[:, :, mid, None] & p_mask_t).sum((0, 1)).float()
            union_vec = (gt_mask_t[:, :, mid, None] | p_mask_t).sum(
                (0, 1)).float() + 1e-5  # add epsilon to avoid divide by zero when us downscaled masks
            iou_vectors.append(inter_vec / union_vec)
            inter_vectors.append(inter_vec)

        iou_matrix = torch.stack(iou_vectors).cpu()

        gt_index, pred_index = linear_sum_assignment(-iou_matrix)

        t3 = time.time()

        # get result array
        result_array, gt_pred_cover = self.fill_result_array(iou_matrix, gt_index, pred_index, p_scores, p_mask_t,
                                                             p_sizes, p_class, gt_mask_t, gt_sizes, gt_class)

        t4 = time.time()

        # Calculate hausdorff distance if required
        if self.calculate_hausdorff:
            # Iterate over result array and calculate hausdorff distance if prediction is TP
            for i in range(result_array.shape[0]):
                if result_array[i, RA_TP_DET_FLAG]:
                    # Get matching gt_mask and pad it with zeros (this is to allow shapes on edges to be closed contours)
                    gt_mask_h = gt_mask_t[:, :, int(result_array[i, RA_GT_MATCHED])]
                    gt_mask_h = pad_mask(gt_mask_h.cpu())

                    # Get predicted mask and pad it with zeros (this is to allow shapes on edges to be closed contours)
                    pred_mask_h = p_mask_t[:, :, i]
                    pred_mask_h = pad_mask(pred_mask_h.cpu())

                    # Get HD distance
                    hd_distance = calculate_hausdorff_distance(pred_mask_h, gt_mask_h)
                    hd_distance_normalized = (HD_DIST_NORMALIZE_CLIPPING - min(HD_DIST_NORMALIZE_CLIPPING,
                                                                               hd_distance)) / HD_DIST_NORMALIZE_CLIPPING
                    result_array[i, RA_HAUSDORFF_DISTANCE_SCORE] = hd_distance_normalized

        result_list.append((result_array, gt_pred_cover))

        t5 = time.time()

        return result_list, t5 - t4, t4 - t3, t3 - t2, t2 - t1, t1 - start

    def get_image_with_gt(self, index):
        """
        This function should be implemented in the specific class according to the framework type.
        :param index: index of image in dataset
        :return: should return 2 numpy arrays.
                    1. gt_class_id- shape (Num of GT objects, 1)
                    2. gt_mask- shape (Num of GT objects, W, H)
        """
        raise NotImplementedError

    def get_prediction(self, image, depth):
        """
        This function should be implemented in the specific class according to the framework type.
        :param image: image to be sent to model, should match the type of image used by the framework
        :return: should return 2 numpy arrays.
                    1. class_id- shape (Num of detected objects, 1)
                    2. mask- shape (Num of detected objects, W, H)
                    3. scores- shape (Num of detected objects, 1)
        """
        raise NotImplementedError


class MaskStatisticsTorch(MaskStatistics):
    """
    Class for holding statistics regarding instance segmentation for pytorch implementation.
    """

    # TODO: add calc housdorf field
    def __init__(self, iou_threshold, dataset, model, path=None, thresh=0.5, device=None, classes=1,
                 size_thresh=(32 ** 2, 96 ** 2), singe_class_mode=False, scaling_factor=1):
        """
        :param iou_threshold: float
        :param dataset: dataset
        :param model: nn.module
        :param path: string path. if given ist the the path for saving a recall precision graph.
        :param thresh: float, thresh of the prediction mask
        :param device: torch.device (cpu\cuda)
        :param classes: int, num of classes, not including background
        :param size_thresh: tuple, size thresholds (in pixels)
        :param singe_class_mode:
        :param scaling_factor: float (compress images and prediction to save memory)
        """
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.thresh = thresh
        self.device = device
        self.singe_class_mode = singe_class_mode
        if self.singe_class_mode:
            classes = 1
        super().__init__(iou_threshold, list(range(len(dataset))), path, classes=classes, size_thresh=size_thresh,
                         scaling_factor=scaling_factor)

    def get_image_with_gt(self, index):
        """
        This function should be implemented in the specific class according to the framework type.
        :param index: index of image in dataset
        :return: should return 2 numpy arrays and an image
                    1. gt_class_id- shape (Num of GT objects, 1)
                    2. gt_mask- shape (W, H, Num of GT objects)
                    3. image
        """
        data, _ = self.dataset[index]
        image = data["main_input"]
        depth = data.get("additional_input")
        gt_dict = data["targets"]
        gt_class_id = gt_dict['labels']
        gt_area = gt_dict['area']

        if self.scaling_factor != 1:
            _, h, w = image.size()
            h = int(h * self.scaling_factor)
            w = int(w * self.scaling_factor)
            gt_dict['masks'] = F.interpolate(gt_dict['masks'].float().unsqueeze(0), size=(h, w),
                                             mode='bilinear').squeeze(0)
            gt_dict['masks'] = torch.where(gt_dict['masks'] >= 0.5, torch.ones_like(gt_dict['masks']),
                                           torch.zeros_like(gt_dict['masks']))

        gt_mask = gt_dict['masks'].permute(1, 2, 0)

        if self.singe_class_mode:
            gt_class_id = torch.ones_like(gt_class_id)

        return gt_class_id, gt_area, gt_mask, image, depth

    def get_prediction(self, image=None, depth=None):
        """
        This function should be implemented in the specific class according to the framework type.
        :param image: image to be sent to model, should match the type of image used by the framework
        :return: should return 2 numpy arrays.
                    1. class_id- shape (Num of detected objects, 1)
                    2. mask- shape (Num of detected objects, W, H)
                    3. scores- shape (Num of detected objects, 1)
        """

        assert image is not None or depth is not None, "Invalid inputs, both inputs are None"

        with torch.no_grad():
            pred = self.model(
                main_input=[image.to(self.device)] if image is not None else [depth.to(self.device)],
                additional_input=[depth.to(self.device)] if depth is not None else None
            )

        masks = pred[0]['masks'].squeeze(1)

        if self.scaling_factor != 1:
            _, h, w = image.size()
            h = int(h * self.scaling_factor)
            w = int(w * self.scaling_factor)
            masks = F.interpolate(masks.float().unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)

        masks = torch.where(masks >= 0.5, torch.ones_like(masks), torch.zeros_like(masks))
        class_ids = pred[0]['labels']  # tensor of size predx1
        masks = masks.permute(1, 2, 0)  # transfer from torch pred x 1 x H x W to np H x W x pred
        masks = torch.where(masks > self.thresh, torch.ones_like(masks),
                            torch.zeros_like(masks))  # transfer to binary mask
        scores = pred[0]['scores']  # tensor of size pred

        if self.singe_class_mode:
            class_ids = torch.ones_like(class_ids)

        return class_ids.detach().cpu().numpy(), masks.type(torch.cuda.IntTensor), scores.detach().cpu().numpy()


# TODO: implement confusion matrix caltulation in depths models

def get_confusion_matrix(model, dataloader, class_number, iou_thresh, device):
    """
    :param model: nn detection model
    :param dataloader: dataloader
    :param class_number: int, number of classes not including background
    :param iou_thresh: float
    :param device: cuda/cpu
    :return: np confusion matrix
    """
    confusion_matrix = np.zeros((class_number + 1, class_number + 1))

    model.to(device)

    model.eval()

    for img, gt in tqdm(dataloader, position=0):
        img = [i.to(device) for i in img]
        pred = model(img)
        pred = [{k: torch.Tensor.detach(v) for k, v in p.items()} for p in pred]

        for i in range(len(img)):
            image_confusion_matrix = get_image_confusion_matrix(pred[i], gt[i], class_number, iou_thresh, device)

            confusion_matrix += image_confusion_matrix

    return confusion_matrix


def get_image_confusion_matrix(pred, gt, class_number, iou_thresh, device):
    """
    :param pred: dict, prediction of the model
    :param gt: dict, gt of the image
    :param class_number: int, number of classes not including background
    :param iou_thresh: float
    :param device: cuda/cpu
    :return: np confusion matrix for single image prediction
    """

    image_confusion_matrix = np.zeros((class_number + 1, class_number + 1))

    h = gt['masks'][0].squeeze().size(0)
    w = gt['masks'][0].squeeze().size(1)

    ones = torch.ones(h, w).to(device).type(torch.cuda.IntTensor)
    zeros = torch.zeros(h, w).to(device).type(torch.cuda.IntTensor)

    # resize from (n,1,h,w)->(n,h,w)
    pred['masks'] = torch.where(pred['masks'] > 0.5, ones, zeros).resize(pred['masks'].size(0), h, w)

    _, relevant_pred = remove_masks_from_edges(pred['masks'].permute(1, 2, 0))
    _, relevant_gt = remove_masks_from_edges(gt['masks'].permute(1, 2, 0))

    pred['masks'] = pred['masks'][relevant_pred == 1]
    pred['labels'] = pred['labels'][relevant_pred == 1]

    gt['masks'] = gt['masks'][relevant_gt == 1]
    gt['labels'] = gt['labels'][relevant_gt == 1]

    for i in range(pred['masks'].shape[0]):
        pred_mask = pred['masks'][i]
        gt_index = get_correlated_gt_index_for_pred(pred_mask, gt['masks'].to(device).type(torch.cuda.IntTensor),
                                                    iou_thresh)
        gt_class = int(gt['labels'][gt_index]) if gt_index != -1 else 0  # 0 = background
        pred_class = int(pred['labels'][i])

        image_confusion_matrix[pred_class, gt_class] += 1

    return image_confusion_matrix


def get_correlated_gt_index_for_pred(pred_mask, gt_mask, iou_thresh):
    """
    :param pred_mask: tensor, the prediction mask
    :param gt_mask: tenso, the gt masks
    :param iou_thresh: float
    :return: index of the most correlated gt, if not exists , return -1
    """

    intersection = (gt_mask & pred_mask).sum(axis=(1, 2)).float()
    union = (gt_mask | pred_mask).sum(axis=(1, 2)).float()
    iou = intersection / union

    index = torch.argmax(iou)

    if iou[index] < iou_thresh:
        index = -1

    return index


def get_scores_with_differents_thresholds(mst, thresh_idx_list, beta=1, fixed=True):
    """
    calculating statistic for multiple thresh values (per size)
    :param mst: mst object
    :param thresh_idx_list: list of thresh idx (int) per sizes
    :param beta: float, beta factor for f1 score
    :param fixed: bool, indicated if statistic calculated for fixed results
    :return: f1, precision, recall for the model with different thresholds
    """
    gt = mst.get_gt_num()

    tp = 0

    fp = 0

    for i in range(len(thresh_idx_list)):
        tp += mst.get_tp_num(thresh_idx_list[i], size=i, fixed=fixed)

        fp += mst.get_fp_num(thresh_idx_list[i], size=i, fixed=fixed)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (gt + 1e-6)
    f1 = (1 + beta ** 2) * (recall * precision / (beta ** 2 * precision + recall + 1e-6))

    return precision, recall, f1
