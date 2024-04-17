import torch
from .ot import Cost_Gaussian_function, OT

def generate_pseudo_label(bev_cur_ngs, bev_futr_ngs, disp_preds, out_seq_len):
    device = disp_preds.device
    # Initialize with predicted displacement
    reference_initwith_pred = Init_with_prediction(bev_cur_ngs, disp_preds)
    # solve optimal transport problem 
    # distance cost
    thresh = 5 * torch.arange(1, out_seq_len+1).to(device)
    Cost_dist, Support = Cost_Gaussian_function(reference_initwith_pred.transpose(3, 2)*0.25, bev_futr_ngs.transpose(3, 2)*0.25, threshold_2=thresh) #
    Cost = Cost_dist
    # Optimzal transport plan
    T = OT(Cost, epsilon=0.03, OT_iter=4)
    # Hard correspondense matrix
    T_indices = T.max(3).indices
    matrix2 = torch.nn.functional.one_hot(T_indices, num_classes=T.shape[3])
    valid_map = matrix2 * Support
    valid_vector = valid_map.sum(dim=3).to(torch.bool)

    # pseudo label
    nn_bev_futr_ngs = torch.gather(input=bev_futr_ngs, dim=2, index=T_indices[:,:,:,None].repeat(1,1,1,2))
    pseudo_label = nn_bev_futr_ngs*0.25 - bev_cur_ngs[:, None, :, :]*0.25
    pseudo_label = pseudo_label * valid_vector[:, :, :, None]
    # 
    # get dense pseudo motion label map
    disps_map = torch.zeros([pseudo_label.shape[0], out_seq_len, 256, 256, 2], device=device)
    for b in range(pseudo_label.shape[0]):
        disps_map[b][:, bev_cur_ngs[b][:,0], bev_cur_ngs[b][:,1]] = pseudo_label[b]

    return disps_map

def Init_with_prediction(nonground_reference_batch, disp_preds):
    '''
    nonground_reference_batch: (B, N, 2)
    disp_preds: (B, future_frames_num, 2, w, h)
    '''
    batch, sample_num, _ = nonground_reference_batch.shape
    _, future_frames_num, _, _, _ = disp_preds.shape
    # Compute the minimum and maximum voxel coordinates
    nonground_reference_batch = nonground_reference_batch.to(torch.int64)
    batch_indices = torch.arange(batch)[:, None].repeat(1, sample_num).view(-1)
    # get correspondent prediction
    disp_pred_reference = disp_preds[batch_indices, :, :, nonground_reference_batch[:, :, 0].view(-1), nonground_reference_batch[:, :, 1].view(-1)]
    disp_pred_reference = disp_pred_reference.view(batch, sample_num, disp_pred_reference.shape[1], 2)
    disp_pred_reference = disp_pred_reference * 4 # to bev map size
    # add 
    nonground_reference_batch = nonground_reference_batch[:,:,None,:].repeat(1,1,future_frames_num,1)
    nonground_reference_batch = nonground_reference_batch + disp_pred_reference
    nonground_reference_batch = nonground_reference_batch.permute(0, 2, 1, 3).contiguous()

    return nonground_reference_batch


        

                            