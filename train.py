# modified based on 'train_single_seq.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import sys
import torch
import torch.optim as optim
import numpy as np
import time
import os

import argparse
from shutil import copytree, copy
from model import MotionNet
from data.nuscenes_dataloader import NuscenesDataset
from pseudo_utils.generate import generate_pseudo_label
from test import eval_motion_displacement
from utils.loss import compute_and_bp_loss
from utils.misc import AverageMeter, check_folder

# initialize
out_seq_len = 5  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)

pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation

voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

parser = argparse.ArgumentParser()
# data
parser.add_argument('--train_data', default="/bevGS_nuScenes/train", type=str, help='The path to the preprocessed BEV training data')
parser.add_argument('--test_data', default='/bev_nuScenes/val/', type=str, help='The path to the preprocessed BEV validation data')
# training
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=20, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='/logs', help='The path to the output log file')
# parameter
parser.add_argument('--if_cluster', action='store_true', help='If using cluster loss')
parser.add_argument('--if_forward', action='store_true', help='If using forward loss')
parser.add_argument('--if_reverse', action='store_true', help='If using backward loss')
parser.add_argument('--dc', default=3, type=float, help='cluster distance')

args = parser.parse_args()
print(args)

def main():
    start_epoch = 1
    # Whether to log the training information
    if args.log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        model_save_path = check_folder(logger_root)
        model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()
        
        # Logging Evaluation details
        eval_file_name = os.path.join(model_save_path, 'eval.txt')
        eval_saver = open(eval_file_name, 'w')
        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        # Copy the code files as logs
        copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
        copytree('data', os.path.join(model_save_path, 'data'))
        copytree('pseudo_utils', os.path.join(model_save_path, 'pseudo_utils'))
        copytree('utils', os.path.join(model_save_path, 'utils'))
        python_files = [f for f in os.listdir('.') if f.endswith('.py')]
        for f in python_files:
            copy(f, model_save_path)
    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    
    # train dataset
    trainset = NuscenesDataset(dataset_root=args.train_data, split='train', future_frame_skip=0, out_len=out_seq_len)
    trainsampler = torch.utils.data.RandomSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, num_workers=args.nworker, 
                                              sampler=trainsampler, collate_fn=trainset.collate_fn, drop_last=True)
    print("Training dataset size:", len(trainset))
    # test dataset 
    testset = NuscenesDataset(dataset_root=args.test_data, split='test', future_frame_skip=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.nworker)

    model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.5)

    for epoch in range(start_epoch, args.nepoch + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if args.log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()
        
        # train
        model.train()

        running_loss_disp_slow = AverageMeter('Slow', ':.4f')  
        running_loss_disp_fast = AverageMeter('Fast', ':.4f')  
        running_loss_disp_cons = AverageMeter('cluster', ':.4f')  
        running_loss_disp_forward = AverageMeter('forward', ':.4f') 
        running_loss_disp_reverse = AverageMeter('reverse', ':.4f')
        running_loss_disp_total = AverageMeter('total', ':.4f')
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, inverse_inputs, g_indices, ng_indices, bev_cur_g, bev_cur_ng, bev_futr_ng = data # batch, seq, w, h, channel
            # Move to GPU/CPU
            inputs = inputs.to(device)
            inverse_inputs = inverse_inputs.to(device)
            bev_cur_g = bev_cur_g.to(device)
            bev_cur_ng = bev_cur_ng.to(device)
            bev_futr_ng = bev_futr_ng.to(device)
            # reverse input in temporal order
            if args.if_reverse:
                disp_pred_reverse = model(inverse_inputs)
                disp_pred_reverse = disp_pred_reverse.view(args.batch, out_seq_len, disp_pred_reverse.size(-3), disp_pred_reverse.size(-2), disp_pred_reverse.size(-1))
            else:
                disp_pred_reverse = None
            # Make prediction
            disp_pred = model(inputs)
            disp_pred = disp_pred.view(args.batch, out_seq_len, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
            #
            if pred_adj_frame_distance:
                for c in range(1, disp_pred.size(1)):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
                    if args.if_reverse:
                        disp_pred_reverse[:, c, ...] = disp_pred_reverse[:, c, ...] + disp_pred_reverse[:, c - 1, ...]

            # get pseudo labels
            pseudo_label = generate_pseudo_label(bev_cur_ng, bev_futr_ng, disp_pred, out_seq_len=out_seq_len)
            # Compute and back-propagate the losses
            loss, loss_disp_slo, loss_disp_fas, loss_disp_cons, loss_forward, loss_reverse = compute_and_bp_loss(device, ng_indices, g_indices,
                                                                                                                        pseudo_label, disp_pred, disp_pred_reverse, args)
            loss.backward()
            optimizer.step()

            running_loss_disp_slow.update(loss_disp_slo)
            running_loss_disp_fast.update(loss_disp_fas)
            running_loss_disp_cons.update(loss_disp_cons)
            running_loss_disp_forward.update(loss_forward)
            running_loss_disp_reverse.update(loss_reverse)
            running_loss_disp_total.update(loss.item())
            # running_loss_disp_motion.update(loss_motion)
            print("{}\t{}\t {} \n{} \t{} \t{} \tat epoch {}, \titerations {}".format(running_loss_disp_total, running_loss_disp_slow, running_loss_disp_fast,
                                                                            running_loss_disp_cons, running_loss_disp_forward, 
                                                                            running_loss_disp_reverse, epoch, i))
        scheduler.step()
        if args.log:
            saver.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(running_loss_disp_total, running_loss_disp_slow, running_loss_disp_fast, 
                                                        running_loss_disp_cons, running_loss_disp_forward, running_loss_disp_reverse))
            saver.flush()

        # save model
        if args.log and (epoch >= 0):
            eval_saver.write(f'epoch:{epoch}\n')
            eval_saver.flush()
            eval_motion_displacement(device=device, model=model,
                                    saver=eval_saver,
                                    use_adj_frame_pred=pred_adj_frame_distance,
                                    dataloader = testloader)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if args.log:
        saver.close()



if __name__ == "__main__":
    main()
