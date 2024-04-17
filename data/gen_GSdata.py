# modified based on 'gen_data.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import sys
import os
sys.path.append('root_path/SelfMotion')
sys.path.append('root_path/SelfMotion/nuscenes-devkit/python-sdk/')
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
patchwork_module_path = "root_path/patchwork-plusplus/build/python_wrapper"
sys.path.insert(0, patchwork_module_path)
import pypatchworkpp

import numpy as np
import argparse
from data.data_utils import voxelize_occupy

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/nuscene', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/bevGS_nuScenes/train', type=str, help='Directory for saving the generated data')
args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
print("Total number of scenes:", len(nusc.scene))

class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4
             
# Patchwork++ initialization
params = pypatchworkpp.Parameters()
params.verbose = True
PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 1  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 20
    skip_frame = 0
    num_adj_seqs = 1


# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 3  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence


scenes = np.load('data/split.npy', allow_pickle=True).item().get(args.split)
print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))

# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def gen_data():
    res_scenes = list()
    for s in scenes:
        s_id = s.split('_')[1]
        res_scenes.append(int(s_id))

    for scene_idx in res_scenes:
        curr_scene = nusc.scene[scene_idx]

        first_sample_token = curr_scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)
        curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

        save_data_dict_list = list()  # for storing consecutive sequences; the data consists of timestamps, points, etc
        adj_seq_cnt = 0
        save_seq_cnt = 0  # only used for save data file name


        # Iterate each sample data
        print("Processing scene {} ...".format(scene_idx))
        while curr_sample_data['next'] != '':

            # Get the synchronized point clouds
            all_pc, all_times, trans_matrices = \
                LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                                    return_trans_matrix=True,
                                                                    nsweeps_back=nsweeps_back,
                                                                    nsweeps_forward=nsweeps_forward)
            # Store point cloud of each sweep
            pc = all_pc.points
            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)

            # Make sure we have sufficient past and future sweeps
            if num_sweeps != (nsweeps_back + nsweeps_forward):

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

                # Reset
                adj_seq_cnt = 0
                save_data_dict_list = list()
                continue

            # Prepare data dictionary for the next step (ie, generating BEV maps)
            save_data_dict = dict()

            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                save_data_dict['pc_' + str(tid)] = _pc

            save_data_dict['times'] = unique_times
            save_data_dict['num_sweeps'] = num_sweeps
            save_data_dict['trans_matrices'] = trans_matrices

            # Get the synchronized bounding boxes
            # First, we need to iterate all the instances, and then retrieve their corresponding bounding boxe
            save_data_dict_list.append(save_data_dict)

            # Update the counter and save the data if desired (But here we do not want to
            # save the data to disk since it would cost about 2TB space)
            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:

                # ------------------------ Now we generate dense BEV maps ------------------------
                for seq_idx, seq_data_dict in enumerate(save_data_dict_list):
                    dense_bev_data = convert_to_dense_bev(seq_data_dict)
                    sparse_bev_data = convert_to_sparse_bev(dense_bev_data)

                    # save the data
                    # save_directory = check_folder(os.path.join(args.savepath, str(scene_idx) + '_' + str(save_seq_cnt)))
                    save_file_name = os.path.join(args.savepath, str(scene_idx) + '_' + str(save_seq_cnt) + '.npy')
                    np.save(save_file_name, arr=sparse_bev_data)

                    print("  >> Finish sample: {}, sequence {}".format(save_seq_cnt, seq_idx))
                # --------------------------------------------------------------------------------

                save_seq_cnt += 1
                adj_seq_cnt = 0
                save_data_dict_list = list()

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            else:
                flag = False
                for _ in range(skip_frame + 1):
                    if curr_sample_data['next'] != '':
                        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more sample frames
                    break


# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
def convert_to_dense_bev(data_dict):
    num_sweeps = data_dict['num_sweeps']
    times = data_dict['times']

    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc.T)

    # Reorder the pc, and skip sample frames if wanted
    # Currently the past frames in pc_list are stored in the following order [current, current + 1, current + 2, ...]
    # Therefore, we would like to reorder the frames
    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(past_frame_skip + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + future_frame_skip)::(future_frame_skip + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2  # now the order is: [past frames -> current frame -> future frames]

    num_past_pcs = len(tmp_pc_list_1)
    num_future_pcs = len(tmp_pc_list_2)

    # Discretize the input point clouds, and compute the ground-truth displacement vectors
    # The following two variables contain the information for the
    # compact representation of binary voxels, as described in the paper
    voxel_indices_list = list()
    padded_voxel_points_list = list()
    nonground_list = list()
    ground_list = list()

    past_pcs_idx = list(range(num_past_pcs+num_future_pcs))
    past_pcs_idx = past_pcs_idx[-(num_past_frames_for_bev_seq+len(tmp_pc_list_2)):]  # we typically use 5 past frames (including the current one)
    for i in past_pcs_idx:
        res, voxel_indices = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, return_indices=True)
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

        # estimate ground
        PatchworkPLUSPLUS.estimateGround(pc_list[i])# Estimate Ground
        res, voxel_indices = voxelize_occupy(PatchworkPLUSPLUS.getGround(), voxel_size=voxel_size, extents=area_extents, return_indices=True)
        ground_list.append(voxel_indices)

        res, voxel_indices = voxelize_occupy(PatchworkPLUSPLUS.getNonground(), voxel_size=voxel_size, extents=area_extents, return_indices=True)
        nonground_list.append(voxel_indices)

    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool_)

    return voxel_indices_list, padded_voxel_points, ground_list, nonground_list


# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(dense_bev_data):
    save_voxel_indices_list, save_voxel_points, ground_list, nonground_list = dense_bev_data

    save_voxel_dims = save_voxel_points.shape[1:]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = save_voxel_indices_list[i].astype(np.int32)
    
    for i in range(len(ground_list)):
        save_data_dict['g_voxel_indices_' + str(i)] = ground_list[i].astype(np.int32)
    
    for i in range(len(nonground_list)):
        save_data_dict['ng_voxel_indices_' + str(i)] = nonground_list[i].astype(np.int32)

    save_data_dict['3d_dimension'] = save_voxel_dims

    # -------------------------------- Sanity Check --------------------------------

    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict['voxel_indices_' + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=np.bool_)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict


if __name__ == "__main__":
    gen_data()

