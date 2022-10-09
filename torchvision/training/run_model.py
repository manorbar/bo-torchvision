import os
import numpy as np
import json
import time
import pathlib
import pickle
import torch.multiprocessing as mp
import sys
import traceback

# Override torchvision package with our local version by inserting it earlier in the python paths
import torch
import cv2
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from building_statistics_utils import MaskStatisticsTorch
from utils import second_to_time_string, create_image_mask_overlay_inference, load_yaml, collate_fn, collate_dict_fn, \
    torch2numpy_image, numpy2torch_image
from runtime_utils import resume, load_datasets, get_model, output_dict_to_summary_writer, \
    get_optimizer, get_lr_scheduler, setup_run, update_tb_grads_visualization, update_tb_highest_loss_images, \
    convert_buildings_data_to_mrcnn, update_tb_epoch_train_results, update_tb_epoch_val_results, parse_args, assert_args
from matplotlib import pyplot as plot
from tqdm import tqdm
from run_statistics import eval_statistics_wrapper
from threading import Thread


def train_epoch(model, data_loader, device, epoch, optimizer, vis_grads_freq=None, summary_writer=None):
    model.train()
    losses_sums = None
    grads = list()
    total_loss = 0

    dloader = tqdm(data_loader, position=0)
    for i, (data, _) in enumerate(dloader):
        data = convert_buildings_data_to_mrcnn(data, device)
        loss_dict = model(**data)
        losses_sums = {loss_type: (
            losses_sums[loss_type] + loss_dict[loss_type].detach() if i > 0 else loss_dict[loss_type].detach()) for
            loss_type in loss_dict.keys()}
        loss = sum(loss for loss in loss_dict.values())
        total_loss += float(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dloader.set_description(f"Mean train loss: {total_loss / (i + 1)}")
        update_tb_grads_visualization(vis_grads_freq, i, model, grads, summary_writer, epoch)

    # Calculate mean loss for each loss type
    losses_sums = {loss_type: losses_sums_value / (i + 1) for loss_type, losses_sums_value in losses_sums.items()}

    # Calculate mean loss per epoch
    mean_loss = total_loss / (i + 1)

    # Log results to tensorboard
    update_tb_epoch_train_results(summary_writer, {"losses_sums": losses_sums, "mean_loss": mean_loss, "epoch": epoch,
                                                   "opt_state": optimizer.state_dict(),
                                                   "model_named_params": model.named_parameters()})

    return mean_loss


def eval_epoch(model, data_loader, device, epoch, config, summary_writer=None, path_to_save_images=None):
    # Validation batch size = 1

    def eval_validation_loss():
        total_loss, max_loss = [0] * 2
        losses_sums, highest_loss_samples_info_lst = dict(), list()
        model.train()
        valid_len = min(config['validation_params']['steps'], len(data_loader))
        valid_indexes = tqdm(range(valid_len), position=0)
        diter = iter(data_loader)

        for i, _ in enumerate(valid_indexes):
            data, sample_name = next(diter)
            data = convert_buildings_data_to_mrcnn(data, device)
            loss_dict = model(**data)
            losses_sums = {loss_type: (
                losses_sums[loss_type] + loss_dict[loss_type].detach() if i > 0 else loss_dict[loss_type].detach()) for
                loss_type in loss_dict.keys()}
            current_loss = np.sum([float(v.item()) for k, v in loss_dict.items()])
            total_loss += current_loss

            max_loss = update_tb_highest_loss_images(
                model,
                current_loss,
                max_loss,
                epoch,
                data,
                sample_name[0],
                highest_loss_samples_info_lst,
                summary_writer,
                highest_loss_vis_thresh=config["validation_params"]["highest_loss_vis_thresh"] if config[
                    "validation_params"].get(
                    "highest_loss_vis_thresh") is not None else 2
            )
            model.train()
            valid_indexes.set_description(f"Mean validation loss: {total_loss / (i + 1)}")

        # Calculate mean loss for each loss type
        losses_sums = {loss_type: losses_sums_value / (i + 1) for loss_type, losses_sums_value in losses_sums.items()}

        # Log samples' info of samples with highest loss
        summary_writer.add_text("Highest_Validation_Loss/sample_info", str(highest_loss_samples_info_lst),
                                global_step=epoch)

        return losses_sums, total_loss, (i + 1)

    def eval_images_vis():
        model.eval()
        vis_images = []
        valid_indexes = tqdm(range(min(len(data_loader), config['validation_params']['num_images_to_save'])),
                             position=0)
        diter = iter(data_loader)

        for i, _ in enumerate(valid_indexes):
            data, sample_name = next(diter)
            eval_data = convert_buildings_data_to_mrcnn(data, device, False)
            pred = model(**eval_data)
            pred = [{k: torch.Tensor.cpu(v) for k, v in p.items()} for p in pred]
            pred_vis = create_image_mask_overlay_inference(eval_data['main_input'][0], pred[0]["masks"], alpha=0.5,
                                                           thresh=config['validation_params']['pixel_thresh'])

            # Save images to disk
            epoch_path = os.path.join(path_to_save_images, f"epoch_{epoch}")
            if not os.path.isdir(epoch_path):
                pathlib.Path(epoch_path).mkdir(parents=True)
            img_save_path = os.path.join(epoch_path, f"{sample_name[0]}.jpg")
            pred_vis = torch2numpy_image(pred_vis)
            pred_vis = cv2.normalize(pred_vis, np.zeros(pred_vis.shape), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            plot.imsave(img_save_path, pred_vis)
            vis_images.append(numpy2torch_image(pred_vis))

        return vis_images

    # Validation loss
    losses_sums, total_loss, val_len = eval_validation_loss()
    mean_loss = total_loss / val_len

    # Validation inference visualization
    temp_box_score_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = config['validation_params']['box_score_thresh']
    vis_images = eval_images_vis()
    model.roi_heads.score_thresh = temp_box_score_thresh  # Return thresh to its original value

    # Log results to tensorboard
    update_tb_epoch_val_results(summary_writer, {"mean_loss": mean_loss, "epoch": epoch, "losses_sums": losses_sums,
                                                 "vis_images": vis_images})

    return mean_loss


def test_model(model, data_loader, device, config, summary_writer=None, path_to_save_images=None):
    """
    test model by calculating statistics and saving images with inference overlay
    :param model: model to evaluate
    :param data_loader: pytorch validation dataloader
    :param device: device to evaluate on, should be torch.device('cpu') or 'cuda'
    :param config: dict representing json config file
    :param summary_writer: object of SummaryWriter, used to loo to tensorboard. if not given, results will not be logged
    :param path_to_save_images: path to save run on selected images
    """

    # Unpack config
    thresh = config['test_params']['pixel_thresh']
    iou_threshold = config['test_params']['map_iou_threshold']
    box_score_thresh = config['test_params']['box_score_thresh']
    num_images_to_save = config['test_params']['num_images_to_save']

    # evaluate some images for visualization
    model.eval()
    # change thresh for inference
    temp_box_score_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = box_score_thresh
    num_images = 0
    for batch_images, batch_labels in tqdm(data_loader):
        num_images += 1
        # move tensors to GPU
        batch_images = [j.to(device) for j in batch_images]
        pred = model(batch_images)
        # move predictions and image tensors back to cpu
        pred = [{k: torch.Tensor.cpu(v) for k, v in p.items()} for p in pred]
        # overlay the predicted masks on to the image
        overlay = create_image_mask_overlay_inference(batch_images[0], pred[0], alpha=0.5, thresh=thresh)
        # save images to disk if path_to_save_images is given
        images_path = os.path.join(path_to_save_images, "images_overlay")
        if not os.path.isdir(images_path):
            pathlib.Path(images_path).mkdir(parents=True)
        img_save_path = os.path.join(images_path, f"image_{num_images}.png")
        plot.imsave(img_save_path, (overlay.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        # add the overlay to the images array to be logged to tensorboard
        if num_images >= num_images_to_save:
            break

    # return thresh to original thresh
    model.roi_heads.score_thresh = temp_box_score_thresh

    # Calculate statistics
    t0 = time.time()
    mst_path = os.path.join(path_to_save_images, "map.png")
    mst = MaskStatisticsTorch(iou_threshold=iou_threshold, dataset=data_loader.dataset, model=model, device=device,
                              path=mst_path)
    mean_average_precision, fixed_mean_average_precision, f1, conf = mst.get_map()
    precision, recall, fixed_precision, fixed_recall = mst.get_precision_recall(box_score_thresh)
    mean_iou = mst.get_mean_iou(box_score_thresh)
    mean_hausdorff = mst.get_mean_hausdorff_distance(box_score_thresh)
    t1 = time.time()
    print(f"statistics time: {second_to_time_string(t1 - t0)}")

    map_string = f"mean average precision: {mean_average_precision}"
    fixed_map_string = f"fixed mean average precision: {fixed_mean_average_precision}"
    precision_recall_iou_string = f"recall: {recall}; precision: {precision}; mean_iou: {mean_iou}"
    fixed_precision_recall_iou_string = f"fixed recall: {fixed_recall}; fixed precision: {fixed_precision}; mean_iou: {mean_iou}"
    hausdorff_string = f"mean hausdorff distance: {mean_hausdorff}"
    f1_string = f"f1 score for confidence {conf}: {f1}"
    print(map_string)
    print(fixed_map_string)
    print(precision_recall_iou_string)
    print(hausdorff_string)
    print(fixed_precision_recall_iou_string)
    print(f1_string)

    # add text results to summary writer
    if summary_writer:
        summary_writer.add_text("Statistics/map", map_string, 0)
        summary_writer.add_text("Statistics/fixed_map", fixed_map_string, 0)
        summary_writer.add_text("Statistics/precision_recall", precision_recall_iou_string, 0)
        summary_writer.add_text("Statistics/hausdorff_distance", hausdorff_string, 0)
        summary_writer.add_text("Statistics/fixed_precision_recall", fixed_precision_recall_iou_string, 0)
        summary_writer.add_text("Statistics/f1", f1_string, 0)


def train(config):
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    else:
        device = torch.device("cpu")
        print("Using cpu only")

    # Create dataset and transform objects
    print("Creating datasets")
    train_ds, valid_ds = load_datasets(config)

    # Create dataloader objects
    print("Creating dataloaders")
    train_loader = DataLoader(train_ds, batch_size=config['train_params']['batch_size'], collate_fn=collate_dict_fn,
                              shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_dict_fn, shuffle=True, num_workers=4)

    # Get model through models' registry
    model = get_model(config)
    model.to(device)

    print("Setting up optimizer")
    model_params = [p for p in model.parameters()]
    optimizer = get_optimizer(model_params, config)

    print("Setting up lr scheduler")
    lr_scheduler = get_lr_scheduler(optimizer, config)

    if isinstance(config['train_params'].get('resume'), dict):  # Resume training from checkpoint if required
        print("Resuming from previous run")
        run_directory = config['train_params']['resume'].get('run_dir')
        if not os.path.isdir(run_directory):
            raise NotADirectoryError(
                f"Invalid run directory to resume: {config['train_params']['resume'].get('run_dir')}")
        summary_dir, checkpoint_dir, tb_dir, model, optimizer, lr_scheduler, start_epoch = \
            resume(config['train_params']['resume'], model, optimizer, lr_scheduler)
    else:  # Create log and run directories
        start_epoch = 0
        run_directory, summary_dir, checkpoint_dir, tb_dir = setup_run(config)

    # Create tensorboard for train and evaluation
    board_summary_writer = SummaryWriter(log_dir=tb_dir)

    print("Beginning to train")
    train_start = time.time()

    # Lists that hold the epochs' statistics' states, determine for each epoch whether to wait to the statistics' processes to finish
    epochs_log_candidate_list = []

    # Log config to tensorboard
    board_summary_writer.add_text("Config", json.dumps(config))

    for epoch in range(int(start_epoch), int(config['train_params']['epochs'])):
        msg = "training epoch: {} finished, total time: {}, train loss: {}, validation loss: {}"
        print("Training epoch: {},\t experiment_name: {}".format(epoch, config["experiment_name"]))

        # Start epoch timer
        epoch_start = time.time()

        # Flush output to verify that tqdm will continue printing on a clear line
        sys.stdout.flush()

        current_mean_loss = 0
        current_mean_loss = train_epoch(model, train_loader, device, epoch, optimizer,
                                        config["train_params"].get("visualize_gradients_frequency"),
                                        summary_writer=board_summary_writer)

        # evaluate model on validation set, save total loss for comparing checkpoints
        validation_loss = 0
        validation_loss = eval_epoch(model, data_loader=valid_loader, device=device, epoch=epoch,
                                     config=config, summary_writer=board_summary_writer,
                                     path_to_save_images=summary_dir)

        # Save checkpoint according to save checkpoint frequency
        if epoch % config['train_params']['save_checkpoint_frequency'] == 0:
            print(
                f"Reached save checkpoint frequency({config['train_params']['save_checkpoint_frequency']}). saving model")
            save_checkpoint_and_run_statistics(checkpoint_dir, config, current_mean_loss, epoch, model, optimizer,
                                               lr_scheduler,
                                               tb_dir, epochs_log_candidate_list)

        # Advance lr scheduler
        lr_scheduler.step(validation_loss)

        # Stop epoch timer
        epoch_total = time.time() - epoch_start
        print(msg.format(epoch, second_to_time_string(epoch_total), current_mean_loss, validation_loss))

        # Log statistics of epochs which have completed the statistics calculation process
        log_statistics_to_tb(board_summary_writer, tb_dir, epochs_log_candidate_list)

        # Flush output to verify that tqdm will continue printing on a clear line
        sys.stdout.flush()

    # Loop until all statistics have been calculated for all epochs
    while len(epochs_log_candidate_list) > 0:
        print("Waiting for all statistics processes to finish calculations (60 seconds)")
        time.sleep(60)  # Wait for 60 seconds before attempting to log statistics again
        log_statistics_to_tb(board_summary_writer, tb_dir, epochs_log_candidate_list)

    # Closure
    train_end = time.time()
    board_summary_writer.close()
    print("Training finished, total time: {}".format(second_to_time_string(train_end - train_start)))
    return


def log_statistics_to_tb(tb_summary_writer, tb_statistics_dir, epochs_log_candidate_list):
    """
    logs statisitcs to tensorboard
    :param tb_summary_writer: torch.utils.tensorboard.SummaryWriter object
    :param tb_statistics_dir: str, path to statistics directory
    :param epochs_log_candidate_list: list, epochs which still require logging
    """
    # iterate over all epochs and log to tb if they still exist in epochs_stats_list_to_log and have already output a stats pickle file
    epochs_log_candidate_list_copy = list(epochs_log_candidate_list)
    for i, epoch in enumerate(epochs_log_candidate_list_copy):
        stats_pickle_file = os.path.join(tb_statistics_dir, f"stats_epoch_{epoch}.p")
        if os.path.exists(stats_pickle_file):
            with open(stats_pickle_file, 'rb') as handle:
                output_dict = pickle.load(handle)
                output_dict_to_summary_writer(output_dict, tb_summary_writer, epoch)
            epochs_log_candidate_list.pop(i)


def save_checkpoint_and_run_statistics(checkpoint_dir, config, current_mean_loss, epoch, model, optimizer, lr_scheduler,
                                       tb_statistics_dir, epochs_log_candidate_list):
    """
    function saves a checkpoint for a given model and runs evaluation process on it
    :param checkpoint_dir: str, path to dir for saving checkpoints
    :param config: dict, configuration for run
    :param current_mean_loss: float, mean loss for current epoch
    :param epoch: int, epoch number
    :param model: nn.module, model for saveing checkpoint
    :param optimizer: nn.optim, optimizer, saved as part of the checkpoint file
    :param lr_scheduler: nn.optim.lr_scheduler, scheduler, saved as part of the checkpoint file
    :param tb_statistics_dir: str, directory for holding logs and scores for statistics run
    :param epochs_log_candidate_list: list, holding epochs which require to be logged
    """
    # add epoch to epoch_stats_list
    epochs_log_candidate_list.append(epoch)

    # run evaluation process
    print(f"Running evaluation process for epoch {epoch}")

    if config['validation_params']['run_statistics_as_process'] is True:
        # run process
        mp.set_start_method(method='spawn', force=True)
        p = mp.Process(target=eval_statistics_wrapper, args=(
            checkpoint_dir, tb_statistics_dir, config, epoch, optimizer, lr_scheduler, current_mean_loss, None,
            model.state_dict()))
        p.start()

    else:
        # using different thread accelerate the calculation and save 50% of calculation time
        t = Thread(target=eval_statistics_wrapper, args=(
            checkpoint_dir, tb_statistics_dir, config, epoch, optimizer, lr_scheduler, current_mean_loss, model))
        t.start()
        t.join()


def test(config):
    # create test run log directory
    run_directory, summary_dir, _, _, = setup_run(config, mode="test")

    # select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using cuda")
    else:
        device = torch.device("cpu")
        print("using cpu only")

    # create dataset and transform objects
    print("creating datasets")
    train_ds, valid_ds = load_datasets(config)

    # create dataloader objects
    print("creating dataloader")
    valid_loader = DataLoader(valid_ds, collate_fn=collate_fn)

    # get model through registry, resume if required
    print("creating model")
    model = get_model(config)

    # load model from given checkpoint
    print("loading checkpoint")
    if os.path.isfile(config['test_params']['model_state_path']):
        with open(config['test_params']['model_state_path'], "rb") as in_check:
            checkpoint = torch.load(in_check)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise Exception(f"{config['test_params']['model_state_path']} is not a valid path to a checkpoint")

    # move model to device
    model.to(device)

    board_summary_writer = SummaryWriter(log_dir=summary_dir)
    # log the config to the tensorboard
    board_summary_writer.add_text("config", json.dumps(config))

    # test epoch
    test_model(model, valid_loader, device, config, summary_writer=board_summary_writer,
               path_to_save_images=run_directory)

    # close summary writer
    board_summary_writer.close()

    return


def main(args):
    config_dict = load_yaml(args["cfg"])
    try:
        if args["mode"] == "train":
            train(config_dict)
            print("Done Training")
        else:
            test(config_dict)
            print("Done Testing")
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    args = parse_args()
    assert_args(args)
    main(args)
