"""
CoFE 轨迹去噪独立评估脚本

对 GT 轨迹施加模拟噪声，使用 CoFE 修复后与原始 GT 对比，
输出 ADE/FDE 指标以评估 CoFE 的去噪精度。

支持:
    - 多噪声水平测试 (sigma 列表)
    - 多 checkpoint 对比 (--checkpoint 可多次指定或逗号分隔)
    - 未训练 baseline (--checkpoint none)
    - 噪声类型选择 (gaussian / shift)

用法:
    python scripts/eval_cofe.py --dtype test --noise_sigma 0.5 1.0 2.0

    python scripts/eval_cofe.py -c ckpt1.pkl -c ckpt2.pkl --noise_sigma 0.5 1.0

    python scripts/eval_cofe.py -c none,ckpt1.pkl -o results.csv

输出:
    - 终端表格 (OrigADE/FDE vs CorrADE/FDE, 以及提升百分比)
    - 可选 CSV 文件
"""

import argparse
import os
import sys
import time

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cofe import CoFE
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='CoFE 轨迹去噪独立评估')

    parser.add_argument('--data_dir', type=str,
                        default='/home/lbh/PTINet/data/JAAD/')
    parser.add_argument('--dataset', type=str, default='T2FPV')
    parser.add_argument('--dtype', type=str, default='test',
                        choices=['train', 'val', 'test'])

    parser.add_argument('--hist_len', type=int, default=5,
                        help='历史序列长度')
    parser.add_argument('--fut_len', type=int, default=5,
                        help='未来序列长度')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--skip', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--loader_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--hidden_size', type=int, default=96)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--layer_norm', action='store_true', default=False)

    parser.add_argument('-c', '--checkpoint', type=str, action='append',
                        default=None,
                        help='CoFE 权重文件 (.pkl)，可多次指定；"none" 表示未训练 baseline')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='结果 CSV 输出路径')
    parser.add_argument('--noise_sigma', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'shift'])

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ['none']
    expanded = []
    for ckpt in args.checkpoint:
        for c in ckpt.split(','):
            expanded.append(c.strip())
    args.checkpoint = expanded

    return args


def add_noise(traj, sigma, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = torch.randn_like(traj) * sigma
        return traj + noise
    elif noise_type == 'shift':
        shift = (torch.rand(traj.shape[0], 1, traj.shape[2], device=traj.device) - 0.5) * sigma * 4
        return traj + shift


def compute_ade_fde(pred, gt, seq_start_end):
    total_ade = 0.0
    total_fde = 0.0
    total_points = 0
    total_seqs = 0

    for start, end in seq_start_end:
        start, end = start.item(), end.item()
        if end <= start:
            continue
        diffs = torch.sqrt(torch.sum((pred[:, start:end] - gt[:, start:end]) ** 2, dim=-1))
        total_ade += diffs.sum().item()
        total_fde += diffs[-1].sum().item()
        total_points += diffs.numel()
        total_seqs += (end - start)

    ade = total_ade / total_points if total_points > 0 else float('inf')
    fde = total_fde / total_seqs if total_seqs > 0 else float('inf')
    return ade, fde


def load_model_state(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def evaluate_cofe(args):
    print('=' * 70)
    print('CoFE 轨迹去噪评估')
    print(f'噪声类型: {args.noise_type}')
    print(f'噪声水平: {args.noise_sigma}')
    print(f'数据: {args.dtype}')
    print(f'Checkpoints: {args.checkpoint}')
    print('=' * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    dataset = getattr(datasets, args.dataset)(
        data_dir=args.data_dir,
        out_dir='/tmp/cofe_eval',
        dtype=args.dtype,
        input=args.hist_len,
        output=args.fut_len,
        stride=args.stride,
        skip=args.skip,
        from_file=False,
        save=False,
        use_images=False,
        use_attribute=False,
        use_opticalflow=False,
    )

    collate_fn = getattr(dataset, 'collate_fn', None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.loader_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
        prefetch_factor=3 if args.loader_workers > 0 else None,
    )

    all_batches = []
    total_pts_global = 0
    total_seqs_global = 0
    for batch in dataloader:
        hist_all = batch.get('hist_all')
        if hist_all is None:
            continue
        seq_start_end = batch.get('seq_start_end')
        if seq_start_end is not None:
            valid = int(seq_start_end[-1, 1].item())
        else:
            B, _, N, _ = hist_all.shape
            valid = B * N
        total_seqs_global += valid
        total_pts_global += valid * hist_all.shape[1]
        all_batches.append({
            k: v.to(device, non_blocking=True) if (v is not None and torch.is_tensor(v)) else v
            for k, v in batch.items() if k in ('hist_all', 'seq_start_end', 'hist_yaw')
        })

    print(f'样本数: {len(all_batches)}, 总点数: {total_pts_global}, 总序列: {total_seqs_global}')

    all_results = []

    for ckpt_path in args.checkpoint:
        cofe = CoFE(
            input_size=2,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            use_resnet=False,
            no_abs=True,
            idxs=[6, 7],
            dropout=args.dropout,
            layer_norm=args.layer_norm,
        ).to(device)

        ckpt_label = 'untrained'
        if ckpt_path.lower() != 'none':
            if not os.path.exists(ckpt_path):
                print(f'\n[WARN] Checkpoint 不存在，跳过: {ckpt_path}')
                continue
            print(f'\n加载: {ckpt_path}')
            state = load_model_state(ckpt_path, device)
            missing, unexpected = cofe.load_state_dict(state, strict=False)
            if missing:
                print(f'  缺失键: {missing}')
            if unexpected:
                print(f'  多余键: {unexpected}')
            ckpt_label = os.path.basename(ckpt_path).replace('.pkl', '').replace('.pth', '')
        else:
            print(f'\n使用未训练 CoFE (baseline)')

        cofe.eval()

        for sigma in args.noise_sigma:
            total_orig_ade = 0.0
            total_orig_fde = 0.0
            total_corr_ade = 0.0
            total_corr_fde = 0.0
            num_batches = 0
            total_time = 0.0

            for batch in all_batches:
                hist_all = batch['hist_all']
                seq_start_end = batch['seq_start_end']
                hist_yaw = batch['hist_yaw']

                B, T, N_pad, F_dim = hist_all.shape
                hist_3d = hist_all.permute(1, 0, 2, 3).reshape(T, B * N_pad, F_dim)

                if seq_start_end is None:
                    seq_start_end = torch.tensor([[0, B * N_pad]], device=device, dtype=torch.long)

                valid_mask = torch.zeros(B * N_pad, dtype=torch.bool, device=device)
                for s, e in seq_start_end:
                    s, e = s.item(), e.item()
                    valid_mask[s:e] = True

                if not valid_mask.any():
                    continue
                valid_count = int(valid_mask.sum().item())
                gt_abs_all = hist_3d[..., :2]
                gt_abs = gt_abs_all[:, valid_mask, :]

                noisy_abs_all = add_noise(gt_abs_all, sigma, args.noise_type)
                noisy_abs = noisy_abs_all[:, valid_mask, :]

                new_seq_start_end = []
                offset = 0
                for s, e in seq_start_end:
                    s, e = s.item(), e.item()
                    n_valid = int(valid_mask[s:e].sum().item())
                    if n_valid > 0:
                        new_seq_start_end.append([offset, offset + n_valid])
                        offset += n_valid
                new_seq_start_end = torch.tensor(new_seq_start_end, device=device, dtype=torch.long)

                orig_ade, orig_fde = compute_ade_fde(noisy_abs, gt_abs, new_seq_start_end)

                hist_yaw_3d = None
                if F_dim >= 3:
                    hist_yaw_3d = hist_3d[..., 2][:, valid_mask]
                    if hist_yaw_3d.sum() == 0:
                        hist_yaw_3d = None

                with torch.no_grad():
                    t0 = time.time()
                    corrected = cofe.infer_correction(
                        noisy_abs,
                        hist_yaw_pred=hist_yaw_3d,
                        hist_seq_start_end=new_seq_start_end,
                    )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    total_time += time.time() - t0

                corr_ade, corr_fde = compute_ade_fde(corrected, gt_abs, new_seq_start_end)

                batch_pts = valid_count * T
                batch_seqs = valid_count
                total_orig_ade += orig_ade * batch_pts
                total_orig_fde += orig_fde * batch_seqs
                total_corr_ade += corr_ade * batch_pts
                total_corr_fde += corr_fde * batch_seqs
                num_batches += 1

            avg_orig_ade = total_orig_ade / total_pts_global if total_pts_global > 0 else float('inf')
            avg_orig_fde = total_orig_fde / total_seqs_global if total_seqs_global > 0 else float('inf')
            avg_corr_ade = total_corr_ade / total_pts_global if total_pts_global > 0 else float('inf')
            avg_corr_fde = total_corr_fde / total_seqs_global if total_seqs_global > 0 else float('inf')

            improve_ade_pct = (avg_orig_ade - avg_corr_ade) / avg_orig_ade * 100 if avg_orig_ade > 0 else 0
            improve_fde_pct = (avg_orig_fde - avg_corr_fde) / avg_orig_fde * 100 if avg_orig_fde > 0 else 0

            all_results.append({
                'checkpoint': ckpt_label,
                'sigma': sigma,
                'OrigADE': round(avg_orig_ade, 4),
                'OrigFDE': round(avg_orig_fde, 4),
                'CorrADE': round(avg_corr_ade, 4),
                'CorrFDE': round(avg_corr_fde, 4),
                'ADE_improve_pct': round(improve_ade_pct, 1),
                'FDE_improve_pct': round(improve_fde_pct, 1),
                'ms_per_batch': round(total_time / num_batches * 1000, 2) if num_batches > 0 else 0,
            })

            print(f'  [{ckpt_label}] sigma={sigma:.1f}: '
                  f'OrigADE={avg_orig_ade:.4f} OrigFDE={avg_orig_fde:.4f} '
                  f'-> CorrADE={avg_corr_ade:.4f} CorrFDE={avg_corr_fde:.4f} '
                  f'({improve_ade_pct:+.1f}% ADE, {improve_fde_pct:+.1f}% FDE)')

    print('\n' + '=' * 70)
    print('汇总结果')
    print('=' * 70)
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f'\n结果已保存至: {args.output}')

    return df


if __name__ == '__main__':
    evaluate_cofe(parse_args())