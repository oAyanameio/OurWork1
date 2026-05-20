"""
PTINet 分布式训练脚本

该脚本用于训练行人意图预测网络（PTINet），支持多GPU分布式训练模式。
"""

import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score, f1_score
import datetime
import datasets

import model.network_image as network
import utils
from utils import data_loader, calculate_score
from torch.utils.tensorboard import SummaryWriter

import visualization.display as viz

def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train PTINet network')
    
    parser.add_argument('--data_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/JAAD/PN/',
                        required=False, help='数据集目录路径')
    parser.add_argument('--dataset', type=str, 
                        default='pie',
                        required=False, help='数据集名称（jaad/pie）')
    parser.add_argument('--out_dir', type=str, 
                        default='/home/farzeen/work/aa_postdoc/intent/PIE_bbox_image/bounding-box-prediction/output',
                        required=False, help='输出目录路径')  
    
    parser.add_argument('--input', type=int,
                        default=16,
                        required=False, help='输入庍列长度（历史帧数）')
    parser.add_argument('--output', type=int, 
                        default=32,
                        required=False, help='输出庍列长度（预测帧数）')
    parser.add_argument('--stride', type=int, 
                        default=16,
                        required=False, help='采样步长')  
    parser.add_argument('--skip', type=int, default=1, help='帧间隔')  

    parser.add_argument('--dtype', type=str, default='train', help='数据类型')
    parser.add_argument("--from_file", type=bool, default=False, help='是否从文件加载预处理数据')       
    parser.add_argument('--save', type=bool, default=True, help='是否保存训练结果')
    parser.add_argument('--log_name', type=str, default='', help='日志文件夹名称')
    parser.add_argument('--loader_workers', type=int, default=16, help='DataLoader 工作线程数')
    parser.add_argument('--loader_shuffle', type=bool, default=True, help='是否打乱数据顺序')
    parser.add_argument('--pin_memory', type=bool, default=False, help='是否使用 pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='预取因子')

    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='单卡批次大小')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='是否使用学习率调度器')
    parser.add_argument('--local-rank', type=int, default=0, help='本地进程排名')

    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--hardtanh_limit', type=int, default=100, help='HardTanh 激活函数边界')
    parser.add_argument('--use_image', type=bool, default=False,
                        help='使用图像作为输入特征')
    parser.add_argument('--image_network', type=str, default='resnet50',
                        help='选择图像主干网络（clstm/resnet50）')
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='使用属性作为输入特征')
    parser.add_argument('--use_opticalflow', type=bool, default=True,
                        help='使用光流作为输入特征')
    
    args = parser.parse_args()

    return args



def train(args, train_set, val_set):
    print('='*100)
    print('Training ...')
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    dist.init_process_group(backend='nccl')
    torch.manual_seed(0)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    verbose = dist.get_rank() == 0
    
    net = network.PTINet(args).cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    
    train_sampler = DistributedSampler(train_set)
    train_collate = getattr(train_set, 'collate_fn', None)
    dataloader_train = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size,
        pin_memory=args.pin_memory, 
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=train_collate,
    )
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    
    data = []
    best_ade = float('inf')
    writer = SummaryWriter()

    def _maybe_intent_feature(batch):
        for key in ('intent_feature', 'vlm_intent_feature', 'intention_feature'):
            if key in batch:
                return batch[key].cuda(non_blocking=True)
        return None

    for epoch in range(args.n_epochs):
        start = time.time()
        avg_epoch_train_s_loss = 0
        avg_epoch_val_s_loss   = 0
        avg_epoch_train_c_loss = 0
        avg_epoch_val_c_loss   = 0
        avg_epoch_train_t_loss = 0
        avg_epoch_val_v_loss   = 0
        ade  = 0
        fde  = 0
        aiou = 0
        fiou = 0
        avg_acc = 0
        avg_rec = 0
        avg_pre = 0
        mAP = 0
        counter = 0
        for idx, inputs in enumerate(dataloader_train):
            counter += 1
            speed = inputs['speed'].cuda(non_blocking=True)
            future_speed = inputs['future_speed'].cuda(non_blocking=True)
            pos = inputs['pos'].cuda(non_blocking=True)
            future_pos = inputs['future_pos'].cuda(non_blocking=True)
            optical = inputs['optical'].cuda(non_blocking=True)
            ped_behavior = inputs['ped_behavior'].cuda(non_blocking=True)
            images = inputs['image'].cuda(non_blocking=True)
            ped_attribute = inputs['ped_attribute'].cuda(non_blocking=True)
            scene_attribute = inputs['scene_attribute'].cuda(non_blocking=True)
            intent_feature = _maybe_intent_feature(inputs)

            hist_all = inputs.get('hist_all', None)
            if hist_all is not None: hist_all = hist_all.cuda(non_blocking=True)

            hist_seq_start_end = inputs.get('seq_start_end', None)
            if hist_seq_start_end is not None: hist_seq_start_end = hist_seq_start_end.cuda(non_blocking=True)

            hist_resnet = inputs.get('hist_resnet', None)
            if hist_resnet is not None: hist_resnet = hist_resnet.cuda(non_blocking=True)

            ego_idx = inputs.get('ego_idx', None)
            if ego_idx is not None: ego_idx = ego_idx.cuda(non_blocking=True)

            hist_abs_gt = inputs.get('hist_all', None)
            if hist_abs_gt is not None: hist_abs_gt = hist_abs_gt.cuda(non_blocking=True)

            hist_yaw_gt = inputs.get('hist_yaw', None)
            if hist_yaw_gt is not None: hist_yaw_gt = hist_yaw_gt.cuda(non_blocking=True)

            net.zero_grad()
            mloss, cofe_loss_val, speed_preds = net(
                speed=speed,
                pos=pos,
                ped_attribute=ped_attribute,
                ped_behavior=ped_behavior,
                scene_attribute=scene_attribute,
                images=images,
                optical=optical,
                intent_feature=intent_feature,
                average=False,
                hist_all=hist_all,
                hist_seq_start_end=hist_seq_start_end,
                hist_resnet=hist_resnet,
                ego_idx=ego_idx,
                hist_abs_gt=hist_abs_gt,
                hist_yaw_gt=hist_yaw_gt,
            )
            speed_loss = mse(speed_preds, future_speed) / 100
            loss = speed_loss + mloss
            loss.backward()
            optimizer.step()
            avg_epoch_train_s_loss += float(speed_loss)
            avg_epoch_train_c_loss += float(cofe_loss_val)
            avg_epoch_train_t_loss += float(loss)
            torch.cuda.synchronize()

        avg_epoch_train_s_loss /= counter
        avg_epoch_train_c_loss /= counter
        avg_epoch_train_t_loss /= counter
        writer.add_scalar("Loss_speed/train", avg_epoch_train_s_loss, epoch)
        writer.add_scalar("Loss_cofe/train", avg_epoch_train_c_loss, epoch)
        writer.add_scalar("Loss/train", avg_epoch_train_t_loss, epoch)

        val_sampler = DistributedSampler(val_set)
        val_collate = getattr(val_set, 'collate_fn', None)
        dataloader_val = torch.utils.data.DataLoader(
            val_set, 
            batch_size=args.batch_size,
            pin_memory=args.pin_memory, 
            num_workers=0, 
            drop_last=True,
            sampler=val_sampler,
            collate_fn=val_collate,
        )
        counter = 0
        state_preds = []
        state_targets = []
        intent_preds = []
        intent_targets = []
        f1_sc = []
        pre = []
        recall_sc = []
        acc = []

        for idx, val_in in enumerate(dataloader_val):
            counter += 1
            speed = val_in['speed'].cuda(non_blocking=True)
            future_speed = val_in['future_speed'].cuda(non_blocking=True)
            pos = val_in['pos'].cuda(non_blocking=True)
            future_pos = val_in['future_pos'].cuda(non_blocking=True)
            ped_attribute = val_in['ped_attribute'].cuda(non_blocking=True)
            scene_attribute = val_in['scene_attribute'].cuda(non_blocking=True)
            optical = val_in['optical'].cuda(non_blocking=True)
            ped_behavior = val_in['ped_behavior'].cuda(non_blocking=True)
            images = val_in['image'].cuda(non_blocking=True)
            intent_feature = _maybe_intent_feature(val_in)

            hist_all = val_in.get('hist_all', None)
            if hist_all is not None: hist_all = hist_all.cuda(non_blocking=True)

            hist_seq_start_end = val_in.get('seq_start_end', None)
            if hist_seq_start_end is not None: hist_seq_start_end = hist_seq_start_end.cuda(non_blocking=True)

            hist_resnet = val_in.get('hist_resnet', None)
            if hist_resnet is not None: hist_resnet = hist_resnet.cuda(non_blocking=True)

            ego_idx = val_in.get('ego_idx', None)
            if ego_idx is not None: ego_idx = ego_idx.cuda(non_blocking=True)

            hist_abs_gt = val_in.get('hist_all', None)
            if hist_abs_gt is not None: hist_abs_gt = hist_abs_gt.cuda(non_blocking=True)

            hist_yaw_gt = val_in.get('hist_yaw', None)
            if hist_yaw_gt is not None: hist_yaw_gt = hist_yaw_gt.cuda(non_blocking=True)

            with torch.no_grad():
                vloss, _, speed_preds = net(
                    speed=speed, 
                    pos=pos,
                    ped_attribute=ped_attribute,
                    ped_behavior=ped_behavior,
                    scene_attribute=scene_attribute,
                    images=images,
                    optical=optical,
                    intent_feature=intent_feature,
                    average=True,
                    hist_all=hist_all,
                    hist_seq_start_end=hist_seq_start_end,
                    hist_resnet=hist_resnet,
                    ego_idx=ego_idx,
                    hist_abs_gt=hist_abs_gt,
                    hist_yaw_gt=hist_yaw_gt,
                )
                speed_loss_v = mse(speed_preds, future_speed) / 100
                crossing_loss_v = 0.0
                avg_epoch_val_s_loss += float(speed_loss_v)
                avg_epoch_val_c_loss += float(crossing_loss_v)
                avg_epoch_val_v_loss += float(vloss)
                preds_p = utils.speed2pos(speed_preds, pos)
                ade += float(utils.ADE(preds_p, future_pos))
                fde += float(utils.FDE(preds_p, future_pos))
                aiou += float(utils.AIOU(preds_p, future_pos))
                fiou += float(utils.FIOU(preds_p, future_pos))
                torch.cuda.synchronize()
            
        avg_epoch_val_s_loss /= counter
        avg_epoch_val_c_loss /= counter
        ade  /= counter
        fde  /= counter     
        aiou /= counter
        fiou /= counter
        v_loss = avg_epoch_val_s_loss + avg_epoch_val_c_loss + avg_epoch_val_v_loss
        writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
        writer.add_scalar("Loss_crossing/val", avg_epoch_val_c_loss, epoch)
        intent_acc = 0.0
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, \
                    avg_epoch_train_c_loss, avg_epoch_val_c_loss, \
                    ade, fde, aiou, fiou, intent_acc])
        if args.lr_scheduler:
            scheduler.step(avg_epoch_train_t_loss)
        if ade < best_ade:
            best_ade = ade
            file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
            if args.lr_scheduler:
                 modelname = 'model_best' + file + '_scheduler.pkl'
            else:
                modelname = 'model_best' + file + '.pkl'   
            torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        print('e:', epoch, 
             '| ade: %.4f'% ade, 
            '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou,
            '| cofe: %.6f'% avg_epoch_train_c_loss)
   
    df = pd.DataFrame(data, columns=['epoch', 'train_loss_s', 'val_loss_s', 'train_loss_c', 'val_loss_c',
                                     'ade', 'fde', 'aiou', 'fiou', 'intention_acc']) 
    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'data_final' + file + '_scheduler.csv'
            modelname = 'model_final' + file + '_scheduler.pkl'
        else:
            filename = 'data_final' + file + '.csv'
            modelname = 'model_final' + file + '.pkl'
        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
        torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))
    print('='*100)
    print('Done !')


if __name__ == '__main__':
    print("Date and time:", datetime.datetime.now())
    args = parse_args()
    config = parse_config_file('/home/lbh/OurWork1/config.yml')
    if config.get('use_argument_parser') == False:
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])
    print(args)
    train_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='train',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )
    val_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )
    train(args, train_set, val_set)
    