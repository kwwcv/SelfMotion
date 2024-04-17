# modified based on 'nuscenes_dataloader.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
from torch.utils.data import Dataset
import numpy as np
import os
import random
import torch
from data.data_utils import classify_speed_level, voxel2bev

def data_augment(inputs, reverse_inputs, bev_cur_ng, bev_cur_g, bev_futr_ng, factor=0.5):
    '''
    Data augmentation (Inplace)
    '''

    # horizontal flip: id 1
    if random.random() < factor:
        inputs = np.flip(inputs, axis=1)
        reverse_inputs = np.flip(reverse_inputs, axis=1)
        bev_cur_ng[:,0] = 255 - bev_cur_ng[:,0]
        bev_cur_g[:,0] = 255 - bev_cur_g[:,0]
        bev_futr_ng[:,:,0] = 255 - bev_futr_ng[:,:,0] 
        # 255->0
        bev_futr_ng[(bev_futr_ng==(255,0)).all(axis=-1)] = np.array([0,0])
    # vertical flip: id 2
    if random.random() < factor:
        inputs = np.flip(inputs, axis=2)
        reverse_inputs = np.flip(reverse_inputs, axis=2)
        bev_cur_ng[:,1] = 255 - bev_cur_ng[:,1]
        bev_cur_g[:,1] = 255 - bev_cur_g[:,1]
        bev_futr_ng[:,:,1] = 255 - bev_futr_ng[:,:,1] 
        # 255->0
        bev_futr_ng[(bev_futr_ng==(0,255)).all(axis=-1)] = np.array([0,0])

    return inputs, reverse_inputs, bev_cur_ng, bev_cur_g, bev_futr_ng

class NuscenesDataset(Dataset):
    def __init__(self, dataset_root=None, split='train', future_frame_skip=0, out_len=5):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        Input:
        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        """
        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        if split == 'train':
            seq_files = [os.path.join(self.dataset_root, f) for f in os.listdir(self.dataset_root)
            if (os.path.isfile(os.path.join(self.dataset_root, f)))]
            self.out_len = out_len
        else:
            seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                        if os.path.isdir(os.path.join(self.dataset_root, d))]
                
            seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
                if (os.path.isfile(os.path.join(seq_dir, f))) and ('0' in f)]
            
        self.seq_files = seq_files
        self.num_sample_seqs = len(self.seq_files)
        print("The number of sequences: {}".format(self.num_sample_seqs))

        # For training, the size of dataset should be 17065 * 2; for validation: 1719; for testing: 4309
        self.future_frame_skip = future_frame_skip
        self.split = split

    def __len__(self):
        return self.num_sample_seqs

    def samplePcs(self, bev, max_len=3000):
        # 
        pad_coord = np.zeros([1,2])
        if len(bev) < max_len:
            pad_len = max_len - len(bev)
            bev = np.vstack([bev, np.repeat(pad_coord, pad_len, axis=0)])
        elif len(bev) > max_len:
            sampled_index = sorted(random.sample(range(len(bev)), max_len))
            bev = bev[sampled_index]
        return bev
    
    @staticmethod
    def collate_fn(batch):
        '''
        Return:
        padded_voxel_points (B, seq_len, w, h, d): input BEV sequence
        reverse_inputs (B, seq_len, w, h, d): reverse input BEV sequence
        g_indices (N_g, 3): 3D index (batch, w, h) for the total N_g non-empty ground bev cells in the batch
        ng_indices (N_ng, 3): 3D index (batch, w, h) for the total N_ng non-empty non-ground bev cells in the batch
        bev_cur_g_pad (B, N_pad, 2): current ground bev cells
        bev_cur_ng_pad (B, N_pad, 2): current non-ground bev cells
        bev_futr_ng (B, N_pad, future_len, 2): future non-ground bev sequence cells
        '''
        padded_voxel_points, reverse_inputs, bev_cur_g, bev_cur_ng, bev_cur_g_pad, bev_cur_ng_pad, bev_futr_ng= zip(*batch)
        # 
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points, axis=0))
        reverse_inputs = torch.from_numpy(np.stack(reverse_inputs, axis=0))
        bev_cur_g_pad = torch.from_numpy(np.stack(bev_cur_g_pad, axis=0)).to(torch.long)
        bev_cur_ng_pad = torch.from_numpy(np.stack(bev_cur_ng_pad, axis=0)).to(torch.long)
        bev_futr_ng = torch.from_numpy(np.stack(bev_futr_ng, axis=0)).to(torch.long)

        g_indices, ng_indices = list(), list()
        for i in range(len(bev_cur_g)):
            cur_gi = torch.from_numpy(bev_cur_g[i])
            batch_indices = i * torch.ones((cur_gi.shape[0], 1))
            g_indices.append(torch.cat([batch_indices, cur_gi], dim=1))

            cur_ngi = torch.from_numpy(bev_cur_ng[i])
            batch_indices = i * torch.ones((cur_ngi.shape[0], 1))
            ng_indices.append(torch.cat([batch_indices, cur_ngi], dim=1))

        g_indices = torch.cat(g_indices, dim=0).to(torch.long)
        ng_indices = torch.cat(ng_indices, dim=0).to(torch.long)

        return padded_voxel_points, reverse_inputs, g_indices, ng_indices, bev_cur_g_pad, bev_cur_ng_pad, bev_futr_ng
    
    def __getitem__(self, idx):
        '''
        Return:

        For training:
        padded_voxel_points (seq_len, w, h, d): input BEV sequence
        reverse_inputs (seq_len, w, h, d): reverse input BEV sequence
        bev_cur_g (N_1, 2): current ground bev cells
        bev_cur_ng (N_2, 2): current non-ground bev cells
        bev_futr_ng (N_pad, future_len, 2): future non-ground bev sequence cells

        For val/test:
        padded_voxel_points (seq_len, w, h, d): input BEV sequence
        all_disp_field_gt (future_len, w, h, 2): motion ground truth
        all_valid_pixel_maps (future_len, w, h)
        non_empty_map (w, h): 1 for occupied, 0 for empty
        pixel_cat_map (w, h, num_category): category ground truth
        motion_state_gt (w, h, 2): motion_state ground truth, [1,0] for static, [0,1] for moving 
        '''
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()

        dims = gt_dict['3d_dimension']

        # input bev sequence
        padded_voxel_points = list()  
        for i in range(5):
            indices = gt_dict['voxel_indices_' + str(i)]
            indices = indices[(indices[:,0]<dims[0]) & (indices[:,1]<dims[1]) & (indices[:,2]<dims[2])]
            curr_voxels = np.zeros(dims, dtype=np.bool_)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        if self.split == 'train':
            ground_thres = 0
            # reverse input bev sequence
            reverse_inputs = list()
            for i in range(8, 3,-1):
                indices = gt_dict['voxel_indices_' + str(i)]
                indices = indices[(indices[:,0]<dims[0]) & (indices[:,1]<dims[1]) & (indices[:,2]<dims[2])]
                curr_voxels = np.zeros(dims, dtype=np.bool_)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                reverse_inputs.append(curr_voxels)
            reverse_inputs = np.stack(reverse_inputs, 0).astype(np.float32)
            
            vx = gt_dict['ng_voxel_indices_4'] 
            bev_cur_ng = voxel2bev(vx[vx[:, 2] >= ground_thres]).astype(np.int64) # current non-ground bev cells
            g_map = padded_voxel_points[-1].sum(-1)
            g_map[bev_cur_ng[:,0], bev_cur_ng[:,1]] = 0
            bev_cur_g = np.nonzero(g_map)
            bev_cur_g = np.stack(bev_cur_g, axis=-1).astype(np.int64) # current ground bev cells


            future_range = range(5,5+self.out_len)
            bev_futr_ng = list()
            for i in future_range:
                vx = gt_dict[f'ng_voxel_indices_{i}']
                bev_futr_ng.append(self.samplePcs(voxel2bev(vx[vx[:, 2] >= ground_thres]), max_len=2500))
            bev_futr_ng = np.stack(bev_futr_ng, axis=0).astype(np.int64) # future non-ground bev sequence cells

            padded_voxel_points, reverse_inputs, bev_cur_ng, bev_cur_g, bev_futr_ng = \
                data_augment(padded_voxel_points, reverse_inputs, bev_cur_ng, bev_cur_g, bev_futr_ng) # data augmentation

            bev_cur_ng_pad = self.samplePcs(bev_cur_ng, max_len=2500)
            bev_cur_g_pad = self.samplePcs(bev_cur_g, max_len=2500)

            return padded_voxel_points, reverse_inputs, bev_cur_g, \
                bev_cur_ng, bev_cur_g_pad, bev_cur_ng_pad, bev_futr_ng
        else:
            num_future_pcs = gt_dict['num_future_pcs']
            num_past_pcs = gt_dict['num_past_pcs']
            pixel_indices = gt_dict['pixel_indices']

            non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
            non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

            sparse_disp_field_gt = gt_dict['disp_field']
            all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
            all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

            sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
            all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
            all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

            sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
            pixel_cat_map = np.zeros((dims[0], dims[1], 5), dtype=np.float32)
            pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

            motion_state_gt = classify_speed_level(all_disp_field_gt, total_future_sweeps=20,
                                                    future_frame_skip=self.future_frame_skip)
            return padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
                    non_empty_map, pixel_cat_map, num_past_pcs, num_future_pcs, motion_state_gt




