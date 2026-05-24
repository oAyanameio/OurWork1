"""
PTINet 分布式训练核心逻辑

包含 train() 函数，支持多GPU分布式训练模式。
由 scripts/train.py 作为入口调用。
"""

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import pandas as pd
import datetime
import datasets

from models import ptinet as network
from utils.metrics import speed2pos, ADE, FDE
from torch.utils.tensorboard import SummaryWriter


def _maybe_intent_feature(batch):
    for key in ('intent_feature', 'vlm_intent_feature', 'intention_feature'):
        if key in batch:
            return batch[key].cuda(non_blocking=True)
    return None


def _to_cuda(batch, key, non_blocking=True):
    val = batch.get(key)
    if val is not None:
        val = val.cuda(non_blocking=non_blocking)
    return val


def train(args, train_set, val_set):
    print('=' * 100)
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

    if args.cofe_pretrained and os.path.exists(args.cofe_pretrained):
        if verbose:
            print('Loading pretrained CoFE weights from: {}'.format(args.cofe_pretrained))
        cofe_state = torch.load(args.cofe_pretrained, map_location='cuda')
        net.cofe.load_state_dict(cofe_state)
    if args.cofe_frozen:
        if verbose:
            print('Freezing CoFE module (cofe_frozen=True)')
        net.cofe.eval()
        for param in net.cofe.parameters():
            param.requires_grad = False

    net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(val_set)
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

    data = []
    best_ade = float('inf')
    writer = SummaryWriter()

    for epoch in range(args.n_epochs):
        start = time.time()
        avg_epoch_train_s_loss = 0
        avg_epoch_val_s_loss = 0
        avg_epoch_train_c_loss = 0
        avg_epoch_val_c_loss = 0
        avg_epoch_train_t_loss = 0
        avg_epoch_val_v_loss = 0
        ade = 0
        fde = 0
        counter = 0
        for idx, inputs in enumerate(dataloader_train):
            counter += 1
            speed = _to_cuda(inputs, 'speed')
            future_speed = _to_cuda(inputs, 'future_speed')
            pos = _to_cuda(inputs, 'pos')
            future_pos = _to_cuda(inputs, 'future_pos')
            optical = _to_cuda(inputs, 'optical')
            ped_behavior = _to_cuda(inputs, 'ped_behavior')
            images = _to_cuda(inputs, 'image')
            ped_attribute = _to_cuda(inputs, 'ped_attribute')
            scene_attribute = _to_cuda(inputs, 'scene_attribute')
            lcf_features = _to_cuda(inputs, 'lcf_features')
            intent_feature = _maybe_intent_feature(inputs)

            hist_all = _to_cuda(inputs, 'hist_all')
            hist_seq_start_end = _to_cuda(inputs, 'seq_start_end')
            hist_resnet = _to_cuda(inputs, 'hist_resnet')
            ego_idx = _to_cuda(inputs, 'ego_idx')
            hist_abs_gt = _to_cuda(inputs, 'hist_abs_gt') or _to_cuda(inputs, 'hist_all')
            hist_yaw_gt = _to_cuda(inputs, 'hist_yaw')

            net.zero_grad()
            mloss, cofe_loss_val, speed_preds = net(
                speed=speed,
                pos=pos,
                ped_attribute=ped_attribute,
                ped_behavior=ped_behavior,
                scene_attribute=scene_attribute,
                images=images,
                optical=optical,
                lcf_features=lcf_features,
                intent_feature=intent_feature,
                average=False,
                hist_all=hist_all,
                hist_seq_start_end=hist_seq_start_end,
                hist_resnet=hist_resnet,
                ego_idx=ego_idx,
                hist_abs_gt=hist_abs_gt,
                hist_yaw_gt=hist_yaw_gt,
            )
            speed_loss = mse(speed_preds, future_speed) / 100 if future_speed is not None else torch.zeros(1, device='cuda')
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

        val_sampler.set_epoch(epoch)
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

        for idx, val_in in enumerate(dataloader_val):
            counter += 1
            speed = _to_cuda(val_in, 'speed')
            future_speed = _to_cuda(val_in, 'future_speed')
            pos = _to_cuda(val_in, 'pos')
            future_pos = _to_cuda(val_in, 'future_pos')
            ped_attribute = _to_cuda(val_in, 'ped_attribute')
            scene_attribute = _to_cuda(val_in, 'scene_attribute')
            optical = _to_cuda(val_in, 'optical')
            ped_behavior = _to_cuda(val_in, 'ped_behavior')
            images = _to_cuda(val_in, 'image')
            lcf_features = _to_cuda(val_in, 'lcf_features')
            intent_feature = _maybe_intent_feature(val_in)

            hist_all = _to_cuda(val_in, 'hist_all')
            hist_seq_start_end = _to_cuda(val_in, 'seq_start_end')
            hist_resnet = _to_cuda(val_in, 'hist_resnet')
            ego_idx = _to_cuda(val_in, 'ego_idx')
            hist_abs_gt = _to_cuda(val_in, 'hist_abs_gt') or _to_cuda(val_in, 'hist_all')
            hist_yaw_gt = _to_cuda(val_in, 'hist_yaw')

            with torch.no_grad():
                vloss, _, speed_preds = net(
                    speed=speed,
                    pos=pos,
                    ped_attribute=ped_attribute,
                    ped_behavior=ped_behavior,
                    scene_attribute=scene_attribute,
                    images=images,
                    optical=optical,
                    lcf_features=lcf_features,
                    intent_feature=intent_feature,
                    average=True,
                    hist_all=hist_all,
                    hist_seq_start_end=hist_seq_start_end,
                    hist_resnet=hist_resnet,
                    ego_idx=ego_idx,
                    hist_abs_gt=hist_abs_gt,
                    hist_yaw_gt=hist_yaw_gt,
                )
                speed_loss_v = mse(speed_preds, future_speed) / 100 if future_speed is not None else torch.zeros(1, device='cuda')
                avg_epoch_val_s_loss += float(speed_loss_v)
                avg_epoch_val_v_loss += float(vloss)
                if future_pos is not None:
                    preds_p = speed2pos(speed_preds, pos)
                    ade += float(ADE(preds_p, future_pos))
                    fde += float(FDE(preds_p, future_pos))
                torch.cuda.synchronize()

        avg_epoch_val_s_loss /= counter
        ade /= counter
        fde /= counter
        v_loss = avg_epoch_val_s_loss + avg_epoch_val_v_loss
        writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss,
                     avg_epoch_train_c_loss, avg_epoch_val_c_loss,
                     ade, fde])
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
        print("s_Val/s_loss = {:.2f} | v_loss = {:.2f} | ADE = {:.2f} | FDE = {:.2f}".format(
            avg_epoch_val_s_loss, avg_epoch_val_v_loss, ade, fde
        ))

    df = pd.DataFrame(data, columns=['epoch', 'train_loss_s', 'val_loss_s', 'train_loss_c', 'val_loss_c',
                                     'ade', 'fde'])
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
    print('=' * 100)
    print('Done !')