"""
CoFE 第一阶段预训练核心逻辑

仅训练 CoFE 模块的轨迹去噪能力，不训练主网络。
由 scripts/train_cofe_stage1.py 作为入口调用。
"""

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import datasets

from models import ptinet as network
from torch.utils.tensorboard import SummaryWriter


def _resolve_intent(inputs, device):
    for key in ('intent_feature', 'vlm_intent_feature', 'intention_feature'):
        if key in inputs:
            return inputs[key].to(device, non_blocking=True)
    return None


def train_cofe_stage1(args, train_set, val_set):
    print('=' * 100)
    print('CoFE Stage 1: Pre-training CoFE denoising module')
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('CoFE hidden size: ' + str(args.cofe_hidden_size))
    print('CoFE num layers: ' + str(args.cofe_num_layers) + '\n')

    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = network.PTINet(args).to(device)

    cofe_params = list(net.cofe.parameters())
    cofe_param_count = sum(p.numel() for p in cofe_params)
    print('CoFE trainable parameters: {:,}'.format(cofe_param_count))

    optimizer = optim.Adam(cofe_params, lr=args.lr, weight_decay=1e-7)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.5)

    dataloader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        drop_last=True,
        shuffle=args.loader_shuffle,
        prefetch_factor=args.prefetch_factor if args.loader_workers > 0 else None,
    )

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, args.log_name, 'tensorboard'))
    best_val_loss = float('inf')
    data = []

    for epoch in range(args.n_epochs):
        start = time.time()
        avg_epoch_train_loss = 0
        counter = 0

        for idx, inputs in enumerate(dataloader_train):
            counter += 1

            hist_all = inputs.get('hist_all', None)
            if hist_all is None:
                continue
            hist_all = hist_all.to(device, non_blocking=True)

            hist_resnet = inputs.get('hist_resnet', None)
            if hist_resnet is not None:
                hist_resnet = hist_resnet.to(device, non_blocking=True)

            hist_seq_start_end = inputs.get('seq_start_end', None)
            if hist_seq_start_end is not None:
                hist_seq_start_end = hist_seq_start_end.to(device, non_blocking=True)

            hist_abs_gt = inputs.get('hist_abs_gt', inputs.get('hist_all', None))
            if hist_abs_gt is not None:
                hist_abs_gt = hist_abs_gt.to(device, non_blocking=True)

            hist_yaw_gt = inputs.get('hist_yaw', None)
            if hist_yaw_gt is not None:
                hist_yaw_gt = hist_yaw_gt.to(device, non_blocking=True)

            intent_feature = _resolve_intent(inputs, device)

            hist_all_n, hist_resnet_n, seq_start_end_n = net._normalize_fpv_inputs(
                hist_all, hist_resnet, hist_seq_start_end
            )
            T, N, _ = hist_all_n.shape
            hist_abs = hist_all_n[..., :2]
            hist_yaw_fpv = hist_all_n[..., 2]

            if hist_abs_gt is not None:
                gt_all, _, _ = net._normalize_fpv_inputs(
                    hist_abs_gt, None, seq_start_end_n
                )
                gt_abs = gt_all[..., :2]
                gt_yaw = net._resolve_fpv_gt_yaw(hist_yaw_gt, hist_yaw_fpv, T)
            else:
                gt_abs, gt_yaw = hist_abs, hist_yaw_fpv

            net.zero_grad()
            cofe_loss = net.cofe.train_correction(
                hist_abs_gt=gt_abs,
                hist_yaw_gt=gt_yaw,
                hist_abs_pred=hist_abs,
                hist_yaw_pred=hist_yaw_fpv,
                hist_resnet=hist_resnet_n,
                hist_seq_start_end=seq_start_end_n,
                hist_intent=intent_feature,
            )
            cofe_loss.backward()
            optimizer.step()

            avg_epoch_train_loss += float(cofe_loss)

        avg_epoch_train_loss /= max(counter, 1)

        if args.lr_scheduler:
            scheduler.step(avg_epoch_train_loss)

        val_loss = _validate_cofe(net, val_set, device, args)
        data.append([epoch, avg_epoch_train_loss, val_loss])

        writer.add_scalar("CoFE_Loss/train", avg_epoch_train_loss, epoch)
        writer.add_scalar("CoFE_Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.out_dir, args.log_name, args.cofe_save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net.cofe.state_dict(), save_path)
            print('  -> Best model saved (val_loss={:.6f})'.format(val_loss))

        elapsed = time.time() - start
        print('Epoch {:3d}/{:3d} | train_loss={:.6f} | val_loss={:.6f} | time={:.1f}s'.format(
            epoch + 1, args.n_epochs, avg_epoch_train_loss, val_loss, elapsed
        ))

    df = pd.DataFrame(data, columns=['epoch', 'train_loss', 'val_loss'])
    if args.save:
        df.to_csv(os.path.join(args.out_dir, args.log_name, 'cofe_stage1_log.csv'), index=False)
        final_path = os.path.join(args.out_dir, args.log_name, 'cofe_stage1_final.pkl')
        torch.save(net.cofe.state_dict(), final_path)
        print('\nCoFE weights saved to: {}'.format(final_path))

    print('=' * 100)
    print('CoFE Stage 1 Done!')
    writer.close()


def _validate_cofe(net, val_set, device, args):
    net.eval()
    total_loss = 0
    counter = 0

    dataloader_val = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        drop_last=True,
        shuffle=False,
        prefetch_factor=args.prefetch_factor if args.loader_workers > 0 else None,
    )

    with torch.no_grad():
        for inputs in dataloader_val:
            counter += 1

            hist_all = inputs.get('hist_all', None)
            if hist_all is None:
                continue
            hist_all = hist_all.to(device, non_blocking=True)

            hist_resnet = inputs.get('hist_resnet', None)
            if hist_resnet is not None:
                hist_resnet = hist_resnet.to(device, non_blocking=True)

            hist_seq_start_end = inputs.get('seq_start_end', None)
            if hist_seq_start_end is not None:
                hist_seq_start_end = hist_seq_start_end.to(device, non_blocking=True)

            hist_abs_gt = inputs.get('hist_abs_gt', inputs.get('hist_all', None))
            if hist_abs_gt is not None:
                hist_abs_gt = hist_abs_gt.to(device, non_blocking=True)

            hist_yaw_gt = inputs.get('hist_yaw', None)
            if hist_yaw_gt is not None:
                hist_yaw_gt = hist_yaw_gt.to(device, non_blocking=True)

            intent_feature = _resolve_intent(inputs, device)

            hist_all_n, hist_resnet_n, seq_start_end_n = net._normalize_fpv_inputs(
                hist_all, hist_resnet, hist_seq_start_end
            )
            T, N, _ = hist_all_n.shape
            hist_abs = hist_all_n[..., :2]
            hist_yaw_fpv = hist_all_n[..., 2]

            if hist_abs_gt is not None:
                gt_all, _, _ = net._normalize_fpv_inputs(
                    hist_abs_gt, None, seq_start_end_n
                )
                gt_abs = gt_all[..., :2]
                gt_yaw = net._resolve_fpv_gt_yaw(hist_yaw_gt, hist_yaw_fpv, T)
            else:
                gt_abs, gt_yaw = hist_abs, hist_yaw_fpv

            cofe_loss = net.cofe.train_correction(
                hist_abs_gt=gt_abs,
                hist_yaw_gt=gt_yaw,
                hist_abs_pred=hist_abs,
                hist_yaw_pred=hist_yaw_fpv,
                hist_resnet=hist_resnet_n,
                hist_seq_start_end=seq_start_end_n,
                hist_intent=intent_feature,
            )
            total_loss += float(cofe_loss)

    net.train()
    return total_loss / max(counter, 1)