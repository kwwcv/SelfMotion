# modified based on 'test.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import torch
import numpy as np
import argparse
import os

from model import MotionNet
from data.nuscenes_dataloader import NuscenesDataset

def eval_motion_displacement(device, model, saver, use_adj_frame_pred=False,
                             dataloader=None, future_frame_skip=0, num_future_sweeps=20):
    """
    Evaluate the motion prediction results.

    model_path: The path to the trained model
    saver: The path for saving the evaluation results
    use_adj_frame_pred: Whether to predict the relative offset between frames
    future_frame_skip: How many future frames need to be skipped within a contiguous sequence (ie, [1, 2, ... 20])
    """
    model.eval()

    # The speed intervals for grouping the cells
    # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])  # unit: m/s
    # We do not consider > 20m/s, since objects in nuScenes appear inside city and rarely exhibit very high speed
    speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, 3 + 1)  # We evaluate predictions at [0.2, 0.4, ..., 1]s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)  # "20" is because the LIDAR scanner is 20Hz

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)

    for i, data in enumerate(dataloader, 0):
        print(f'Testing: {i}/{len(dataloader)}',end='\r')
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map_gt, past_steps, future_steps, motion_gt = data

        padded_voxel_points = padded_voxel_points.to(device)

        with torch.no_grad():
            disp_pred = model(padded_voxel_points)

            pred_shape = disp_pred.size()
            disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
            disp_pred = disp_pred.contiguous()
            disp_pred = disp_pred.cpu().numpy()

            if use_adj_frame_pred:
                for c in range(1, disp_pred.shape[1]):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]

        static_mask = (np.linalg.norm(disp_pred, ord=2, axis=2) > 0.2)
        disp_pred = disp_pred * static_mask[:,:,None,:,:]
        # Pre-processing
        # disp_pred = disp_pred
        all_disp_field_gt = all_disp_field_gt.numpy()  # (bs, seq, h, w, channel)
        future_steps = future_steps.numpy()[0]

        valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = valid_pixel_maps.numpy()

        all_disp_field_gt = all_disp_field_gt[:, -future_steps:, ]
        all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
        all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

        # -----------------------------------------------------------------------------------
        # Compute the evaluation metrics
        # First, compute the displacement prediction error;
        # Compute the static and moving cell masks, and
        # Iterate through the distance intervals and group the cells based on their speeds;
        upper_thresh = 0.2
        upper_bound = (future_frame_skip + 1) / 20 * upper_thresh

        static_cell_mask = all_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
        moving_cell_mask = np.logical_not(static_cell_mask)

        for j, d in enumerate(distance_intervals):
            for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
                curr_valid_pixel_map = valid_pixel_maps[:, s]

                if j == 0:  # corresponds to static cells
                    curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                else:
                    # We use the displacement between keyframe and the last sample frame as metrics
                    last_gt_norm = all_disp_field_gt_norm[:, -1]
                    mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                    curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                    curr_mask = np.logical_and(curr_mask, moving_cell_mask)

                # Since in nuScenes (with 32-line LiDAR) the points (cells) in the distance are very sparse,
                # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
                border = 8
                roi_mask = np.zeros_like(curr_mask, dtype=np.bool_)
                roi_mask[:, border:-border, border:-border] = True
                curr_mask = np.logical_and(curr_mask, roi_mask)

                cell_idx = np.where(curr_mask == True)

                gt = all_disp_field_gt[:, s]
                pred = disp_pred[:, slot]
                norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

                cell_groups[j][slot].append(norm_error[cell_idx])

    # Compute the statistics
    dump_res = []

    # Compute the statistics of displacement prediction error
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]

        saver.write("--------------------------------------------------------------\n")
        saver.write("For cells within speed range [{}, {}]:\n\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}".\
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            saver.write(msg + "\n")
            saver.flush()

        saver.write("--------------------------------------------------------------\n\n")

        dump_res.append(dump_error + dump_error_quantile_50)

    # Compute the mean classification accuracy for each object category
    saver.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the [val/test] dataset')
    parser.add_argument('-m', '--model', default=None, type=str, help='The path to the trained model')
    parser.add_argument('-l', '--log_path', default=None, type=str, help='The path to the txt file for saving eval results')
    parser.add_argument('-s', '--split', default='test', type=str, help='Which split [val/test]')
    parser.add_argument('-b', '--bs', default=1, type=int, help='Batch size')
    parser.add_argument('-w', '--worker', default=8, type=int, help='The number of workers')
    parser.add_argument('-n', '--net', default='MotionNet', type=str, help='Which network [MotionNet/MotionNetMGDA]')
    parser.add_argument('-a', '--adj', action='store_false', help='Whether predict the relative offset between frames')
    parser.add_argument('-j', '--jitter', action='store_false', help='Whether to apply jitter suppression')

    args = parser.parse_args()
    print(args)

    # Datasets
    testset = NuscenesDataset(dataset_root=args.data, split='test', future_frame_skip=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.worker)
    # model initialize
    device = 'cuda'
    model = MotionNet(out_seq_len=5, motion_category_num=2, height_feat_size=13)
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    # Logging Evaluation details
    eval_file_name = os.path.join(args.log_path, './eval.txt')
    eval_saver = open(eval_file_name, 'w')
    eval_motion_displacement(device, model, saver=eval_saver, use_adj_frame_pred=True, dataloader=testloader)
