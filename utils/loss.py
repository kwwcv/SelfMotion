import torch
import torch.nn.functional as F
import numpy as np

def compute_and_bp_loss(device, ng_indices, g_indices, disp_gt, disp_pred, disp_pred_reverse, args):
    '''
    Input:
    g_indices (N_g, 3): 3D index (batch, w, h) for the total N_g non-empty ground bev cells in the batch
    ng_indices (N_ng, 3): 3D index (batch, w, h) for the total N_ng non-empty non-ground bev cells in the batch
    disp_gt (batch, future_len, w, h, 2): motion displacement peseudo label map
    disp_pred (batch, future_len, 2, w, h): forward motion prediction
    disp_pred_reverse (batch, future_len, 2, w, h): backward motion prediction
    '''
    g_factor = 1.0

    ng_batch_ind, ng_x_ind, ng_y_ind = ng_indices[:, 0], ng_indices[:, 1], ng_indices[:, 2]
    g_batch_ind, g_x_ind, g_y_ind = g_indices[:, 0], g_indices[:, 1], g_indices[:, 2]

    disp_pred_valid_ng = disp_pred[ng_batch_ind, :, :, ng_x_ind, ng_y_ind]
    disp_pred_valid_g = disp_pred[g_batch_ind, :, :, g_x_ind, g_y_ind]

    disp_gt_valid_ng = disp_gt[ng_batch_ind, :, ng_x_ind, ng_y_ind, :]
    disp_gt_valid_g = disp_gt[g_batch_ind, :, g_x_ind, g_y_ind, :]

    if args.if_reverse:
        disp_pred_valid_reverse_ng = disp_pred_reverse[ng_batch_ind, :, :, ng_x_ind, ng_y_ind]
        disp_pred_valid_reverse_g = disp_pred_reverse[g_batch_ind, :, :, g_x_ind, g_y_ind]

    # weighted different speed level
    disp_gt_valid_norm = disp_gt_valid_ng[:,-1,:].norm(dim=-1) # get the dicplacement after 1s
    slow_gt_index = (disp_gt_valid_norm < 5)
    fast_gt_index = (disp_gt_valid_norm >= 5)

    loss_disp_slow, loss_disp_fast = 0, 0
    loss_disp_ng_slow = F.smooth_l1_loss(disp_gt_valid_ng[slow_gt_index], disp_pred_valid_ng[slow_gt_index], reduction='sum')
    loss_disp_ng_fast = F.smooth_l1_loss(disp_gt_valid_ng[fast_gt_index], disp_pred_valid_ng[fast_gt_index], reduction='sum')
    loss_disp_slow += loss_disp_ng_slow
    loss_disp_fast += loss_disp_ng_fast

    loss_disp_g = F.smooth_l1_loss(disp_gt_valid_g, disp_pred_valid_g, reduction='mean')

    # supervise backward motion for faster converge
    if args.if_reverse:
        loss_disp_ng_reverse_slow = F.smooth_l1_loss(-disp_gt_valid_ng[slow_gt_index], disp_pred_valid_reverse_ng[slow_gt_index], reduction='sum')
        loss_disp_ng_reverse_fast = F.smooth_l1_loss(-disp_gt_valid_ng[fast_gt_index], disp_pred_valid_reverse_ng[fast_gt_index], reduction='sum')
        loss_disp_slow += loss_disp_ng_reverse_slow
        loss_disp_fast += loss_disp_ng_reverse_fast

        loss_disp_g_reverse = F.smooth_l1_loss(-disp_gt_valid_g, disp_pred_valid_reverse_g, reduction='mean')
        loss_disp_g = loss_disp_g + loss_disp_g_reverse

    loss_disp_ng = loss_disp_slow + loss_disp_fast * 3
    loss_disp_ng = loss_disp_ng / len(disp_gt_valid_norm)

    loss_disp = loss_disp_g * g_factor + loss_disp_ng
    loss = loss_disp

    # spatial cluster consistency loss
    if args.if_cluster:
        loss_cons = torch.tensor(0.0, device=device)
        for batch, disp_pred_batch in enumerate(disp_pred):
            loss_cons_batch = 0
            ng_ind = ng_indices[ng_batch_ind == batch][:, 1:]
            fast_ng_ind = np.nonzero((disp_pred_batch[-1].norm(dim=0)[ng_ind[:,0], ng_ind[:,1]] >= disp_pred_batch.shape[0]).cpu().numpy())[0]
            if len(fast_ng_ind) > 0:
                # clustering
                instances_indice_list = cluster(ng_ind.cpu().numpy(), fast_ng_ind, max_dist=args.dc)
                for instance_ind in instances_indice_list:
                    ins_disp_pred = disp_pred_batch[:, :, instance_ind[:,0], instance_ind[:,1]]
                    loss_cons_batch += F.l1_loss(ins_disp_pred[:,:,:,None].repeat(1,1,1,len(instance_ind)), 
                                                ins_disp_pred[:,:,None,:].repeat(1,1,len(instance_ind),1), reduction='mean')
                loss_cons += loss_cons_batch / (len(instances_indice_list)+1e-6) 
        loss_cons = loss_cons / len(disp_pred)
        loss += loss_cons * 0.05
    else:
        loss_cons = torch.tensor(0)
        
    # forward consistency loss
    if args.if_forward:
        future_len = torch.arange(1, disp_pred_valid_ng.shape[1]+1).to(device)
        loss_forward_ng = F.smooth_l1_loss(disp_pred_valid_ng[:,1:,:] / future_len[None,1:,None], disp_pred_valid_ng[:,:-1,:] / future_len[None,:-1,None])
        loss_forward_g = F.smooth_l1_loss(disp_pred_valid_g[:,1:,:] / future_len[None,1:,None], disp_pred_valid_g[:,:-1,:] / future_len[None,:-1,None])
        loss_forward = loss_forward_ng + loss_forward_g * g_factor
        loss += loss_forward * 0.1
    else:
        loss_forward = torch.tensor(0)

    # backward consistency loss
    if args.if_reverse:
        uncertain_weight = torch.arange(1, disp_pred.shape[1]+1)
        uncertain_weight = torch.exp(-uncertain_weight / 10).to(device)
        loss_reverse_ng = F.smooth_l1_loss(disp_pred_valid_ng, -disp_pred_valid_reverse_ng, reduction='none')
        loss_reverse_g = F.smooth_l1_loss(disp_pred_valid_g, -disp_pred_valid_reverse_g, reduction = 'none')
        loss_reverse_ng = torch.mean(loss_reverse_ng * uncertain_weight[None,:,None])
        loss_reverse_g = torch.mean(loss_reverse_g * uncertain_weight[None,:,None])

        loss_reverse = loss_reverse_g * g_factor + loss_reverse_ng
        loss += loss_reverse
    else:
        loss_reverse = torch.tensor(0)

    return loss, loss_disp_g.item(), loss_disp_ng.item(), loss_cons.item(), loss_forward.item(), loss_reverse.item()

def cluster(points, fast_ind, max_dist=3):
    # distance map
    distance_matrix = np.abs(points[:,None,:] - points[None,:, :])
    distance_matrix = distance_matrix[:,:,0] + distance_matrix[:,:,1]
    distance_matrix_mask = (distance_matrix < max_dist)
    points_ids = np.linspace(0, len(points)-1, len(points), dtype=np.compat.long)

    instances_list = []
    points_ids_pop_list = fast_ind.tolist()
    while len(points_ids_pop_list) > 0:
        i = points_ids_pop_list.pop()
        point_save_list = [i]
        point_pop_list = [i]
        while len(point_pop_list) > 0:
            distance_masks = distance_matrix_mask[point_pop_list]
            neighbor_ids_list = np.repeat(points_ids[None, :], repeats=len(distance_masks), axis=0)[distance_masks]
            neighbor_ids_list = np.unique(neighbor_ids_list).tolist()
            # difference
            difference_ids_set = set(neighbor_ids_list).difference(set(point_save_list)) # in neighbor_ids_list while not in point_save_list
            # join
            point_pop_list = list(difference_ids_set)
            point_save_list = list(set(point_save_list).union(difference_ids_set))

        instances_list.append(point_save_list)
        points_ids_pop_list = list(set(points_ids_pop_list).difference(set(point_save_list)))
    
    # filter intance with few points
    instances_indices = [points[np.array(lis)] for lis in instances_list if len(lis) >= 5]
    return instances_indices