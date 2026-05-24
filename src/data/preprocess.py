"""
T2FPV 数据集预处理脚本

将 T2FPV 原始 CSV 检测数据预处理为 PTINet 模型可直接使用的训练格式。
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'logs', 'preprocess_t2fpv.log'
        ))
    ]
)
logger = logging.getLogger(__name__)

SCENE_TO_FOLD = {
    'biwi_eth': 'eth',
    'biwi_hotel': 'hotel',
    'students001': 'univ',
    'students003': 'univ',
    'crowds_zara01': 'zara1',
    'crowds_zara02': 'zara2',
    'crowds_zara03': 'zara1',
    'uni_examples': 'univ',
}

ALL_SCENES = [
    'biwi_eth', 'biwi_hotel', 'crowds_zara01', 'crowds_zara02',
    'crowds_zara03', 'students001', 'students003', 'uni_examples',
]

FOLD_TO_SCENES: Dict[str, List[str]] = {
    'eth': ['biwi_eth'],
    'hotel': ['biwi_hotel'],
    'univ': ['students001', 'students003', 'uni_examples'],
    'zara1': ['crowds_zara01', 'crowds_zara03'],
    'zara2': ['crowds_zara02'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='T2FPV 数据集预处理脚本'
    )

    parser.add_argument('--data_root', type=str, default=None,
                        help='FPVDataset 目录路径')
    parser.add_argument('--gt_dets_rel', type=str, default='gt_dets')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: data/processed)')
    parser.add_argument('--output_prefix', type=str, default='t2fpv')

    parser.add_argument('--hist_len', type=int, default=8,
                        help='历史帧数')
    parser.add_argument('--fut_len', type=int, default=12,
                        help='未来帧数')
    parser.add_argument('--frame_skip', type=int, default=10,
                        help='帧采样间隔')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑动窗口步长')
    parser.add_argument('--min_agents', type=int, default=2,
                        help='每个场景最小行人数量')

    parser.add_argument('--scenes', type=str, nargs='+', default=None)
    parser.add_argument('--folds', type=str, nargs='+', default=None)

    parser.add_argument('--save_format', type=str, choices=['pt', 'npy'], default='pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_skip', dest='skip_if_exists', action='store_false',
                        default=True)

    return parser.parse_args()


def resolve_paths(args) -> dict:
    data_root = args.data_root
    if data_root is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'FPVDataset'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'T2FPV-ow', 'FPVDataset'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'T2FPV', 'data', 'FPVDataset'),
        ]
        for c in candidates:
            abs_c = os.path.abspath(c)
            if os.path.isdir(abs_c):
                data_root = abs_c
                logger.info(f"自动检测到数据根目录: {data_root}")
                break

    if data_root is None:
        logger.error(
            "未找到 FPVDataset。请使用 --data_root 指定路径。\n"
            "候选路径:\n" + "\n".join(f"  - {c}" for c in candidates)
        )
        sys.exit(1)

    data_root = os.path.abspath(data_root)
    gt_dets_path = os.path.join(data_root, args.gt_dets_rel)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'processed'
        )
    os.makedirs(output_dir, exist_ok=True)

    return {
        'data_root': data_root,
        'gt_dets': gt_dets_path,
        'output': output_dir,
    }


def determine_scenes(args, paths: dict) -> List[str]:
    if args.scenes:
        return [s for s in args.scenes if s in ALL_SCENES]
    if args.folds:
        scenes = []
        for fold in args.folds:
            scenes.extend(FOLD_TO_SCENES.get(fold, []))
        return scenes
    available = []
    for scene in ALL_SCENES:
        if os.path.isdir(os.path.join(paths['gt_dets'], scene)):
            available.append(scene)
    return available


def load_scene_data(scene_dir: str) -> pd.DataFrame:
    all_dfs = []
    for fname in sorted(os.listdir(scene_dir)):
        if not fname.endswith('_dets.csv'):
            continue
        fpath = os.path.join(scene_dir, fname)
        try:
            agent_id = int(fname.split('agent')[1].split('_')[0])
        except (IndexError, ValueError):
            continue

        df = pd.read_csv(fpath)
        if 'x_w' not in df.columns:
            continue
        df['agent_id'] = agent_id
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def build_scenes_from_dets(
    scene_df: pd.DataFrame,
    hist_len: int,
    fut_len: int,
    frame_skip: int,
    stride: int,
    min_agents: int,
) -> List[dict]:
    full_len = hist_len + fut_len
    if scene_df.empty:
        return []

    samples = []
    scene_starts = sorted(scene_df.scene_start.unique())

    for ss in scene_starts:
        ss_df = scene_df[scene_df.scene_start == ss].copy()
        ss_df = ss_df.sort_values('frame_id')

        pedestrians = []
        for (aid, did), ped_df in ss_df.groupby(['agent_id', 'det_id']):
            ped_df = ped_df.drop_duplicates(subset='frame_id').sort_values('frame_id')
            frames = ped_df['frame_id'].values
            if len(frames) < full_len:
                continue
            pedestrians.append({
                'agent_id': int(aid),
                'det_id': int(did),
                'df': ped_df,
                'frame_min': frames.min(),
                'frame_max': frames.max(),
            })

        if len(pedestrians) < min_agents:
            continue

        ped_frame_min = max(p['frame_min'] for p in pedestrians)
        ped_frame_max = min(p['frame_max'] for p in pedestrians)

        if ped_frame_min > ped_frame_max or (ped_frame_max - ped_frame_min) < full_len * frame_skip:
            continue

        for frame_start in range(int(ped_frame_min), int(ped_frame_max) + 1, stride * frame_skip):
            end_frame = frame_start + full_len * frame_skip
            if end_frame > ped_frame_max:
                break

            sampled_frames = list(range(frame_start, end_frame, frame_skip))
            frame_ids_hist = sampled_frames[:hist_len]
            frame_ids_fut = sampled_frames[hist_len:]

            scene_agents = []
            for ped in pedestrians:
                ped_df = ped['df']
                det_in_window = ped_df[ped_df.frame_id.isin(sampled_frames)]

                if len(det_in_window) < full_len:
                    continue

                det_in_window = det_in_window.set_index('frame_id')
                det_in_window = det_in_window.reindex(sampled_frames).dropna()
                det_in_window = det_in_window.reset_index()

                if len(det_in_window) < full_len:
                    continue

                hist_det = det_in_window.iloc[:hist_len]
                fut_det = det_in_window.iloc[hist_len:]

                hist_pos = np.column_stack([
                    hist_det['x_w'].values.astype(np.float32),
                    hist_det['z_w'].values.astype(np.float32),
                ])
                hist_yaw = hist_det['yaw_w'].values.astype(np.float32)
                hist_abs = np.column_stack([hist_pos, hist_yaw])

                fut_pos = np.column_stack([
                    fut_det['x_w'].values.astype(np.float32),
                    fut_det['z_w'].values.astype(np.float32),
                ])

                scene_agents.append({
                    'agent_id': int(ped['agent_id']),
                    'det_id': int(ped['det_id']),
                    'hist_abs': hist_abs,
                    'fut_pos': fut_pos,
                })

            if len(scene_agents) >= min_agents:
                samples.append({
                    'agents': scene_agents,
                    'frame_ids_hist': frame_ids_hist,
                    'frame_ids_fut': frame_ids_fut,
                })

    return samples


def compute_speed(pos_seq: np.ndarray) -> np.ndarray:
    speed = np.zeros_like(pos_seq)
    if len(pos_seq) > 1:
        speed[1:] = pos_seq[1:] - pos_seq[:-1]
        speed[0] = speed[1]
    return speed


def build_sample_dict(
    scene_samples: List[dict],
    hist_len: int,
    fut_len: int,
    scene_name: str = None,
) -> dict:
    n_agents = len(scene_samples)

    hist_abs = np.stack([s['hist_abs'] for s in scene_samples], axis=1)
    hist_pos = hist_abs[:, :, :2]
    hist_yaw = hist_abs[:, :, 2]
    fut_pos = np.stack([s['fut_pos'] for s in scene_samples], axis=1)

    hist_speed = np.zeros_like(hist_pos)
    for i in range(n_agents):
        hist_speed[:, i, :] = compute_speed(hist_pos[:, i, :])

    fut_speed = np.zeros_like(fut_pos)
    for i in range(n_agents):
        fut_speed[:, i, :] = compute_speed(fut_pos[:, i, :])

    pos = hist_pos.transpose(1, 0, 2)
    speed = hist_speed.transpose(1, 0, 2)
    future_pos = fut_pos.transpose(1, 0, 2)
    future_speed = fut_speed.transpose(1, 0, 2)
    hist_all = hist_abs.transpose(1, 0, 2)
    seq_start_end = np.array([[0, n_agents]], dtype=np.int64)
    ped_behavior = np.zeros((n_agents, hist_len, 3), dtype=np.float32)

    sample = {
        'pos': pos,
        'speed': speed,
        'future_pos': future_pos,
        'future_speed': future_speed,
        'ped_behavior': ped_behavior,
        'hist_all': hist_all,
        'seq_start_end': seq_start_end,
        'hist_yaw': hist_yaw.transpose(1, 0),
        # 供 extract_intent.py 定位 FPV 图像（imgs/{scene_name}/agent{id}_seg/idx*.jpg）
        'agent_ids': np.array([s['agent_id'] for s in scene_samples], dtype=np.int64),
    }
    if scene_name is not None:
        sample['scene_name'] = scene_name
    return sample


def process_data(args, paths: dict) -> Dict[str, list]:
    scenes = determine_scenes(args, paths)
    logger.info(f"处理场景: {scenes}")

    per_fold: Dict[str, Dict[str, list]] = {}

    for scene_name in scenes:
        scene_dir = os.path.join(paths['gt_dets'], scene_name)
        if not os.path.isdir(scene_dir):
            logger.warning(f"场景目录不存在: {scene_dir}")
            continue

        logger.info(f"加载 {scene_name} 检测数据...")
        scene_df = load_scene_data(scene_dir)
        logger.info(f"  共 {len(scene_df)} 行检测数据, {scene_df.det_id.nunique()} 个不同det_id")

        fold = SCENE_TO_FOLD.get(scene_name, scene_name)
        if fold not in per_fold:
            per_fold[fold] = {}

        all_scene_samples = build_scenes_from_dets(
            scene_df=scene_df,
            hist_len=args.hist_len,
            fut_len=args.fut_len,
            frame_skip=args.frame_skip,
            stride=args.stride,
            min_agents=args.min_agents,
        )
        logger.info(f"  构建 {len(all_scene_samples)} 个场景")

        if not all_scene_samples:
            continue

        np.random.shuffle(all_scene_samples)
        n = len(all_scene_samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        split_map = {
            'train': all_scene_samples[:n_train],
            'val': all_scene_samples[n_train:n_train + n_val],
            'test': all_scene_samples[n_train + n_val:],
        }

        for split_name, scene_list in split_map.items():
            if split_name not in per_fold[fold]:
                per_fold[fold][split_name] = []
            per_fold[fold][split_name].extend(scene_list)

        logger.info(f"    训练: {len(split_map['train'])}, "
                    f"验证: {len(split_map['val'])}, "
                    f"测试: {len(split_map['test'])}")

    combined: Dict[str, list] = {}
    for fold, splits in per_fold.items():
        for split_name, scene_list in splits.items():
            if split_name not in combined:
                combined[split_name] = []
            for scene in scene_list:
                sample = build_sample_dict(
                    scene['agents'], args.hist_len, args.fut_len, scene_name=scene_name
                )
                sample['frame_ids_hist'] = np.array(scene['frame_ids_hist'], dtype=np.int64)
                sample['frame_ids_fut'] = np.array(scene['frame_ids_fut'], dtype=np.int64)
                combined[split_name].append(sample)

    return combined


def print_statistics(data: Dict[str, list]):
    print("\n" + "=" * 70)
    print("数据集统计报告")
    print("=" * 70)
    for split_name, samples in data.items():
        n = len(samples)
        if n == 0:
            print(f"\n{split_name}: 0 样本")
            continue
        n_agents = sum(s['pos'].shape[0] for s in samples)
        avg = n_agents / n
        first = samples[0]
        print(f"\n{split_name.upper()}:")
        print(f"  样本数: {n}")
        print(f"  Agent总数: {n_agents}")
        print(f"  平均Agent/场景: {avg:.2f}")
        print(f"  pos维度: {tuple(first['pos'].shape)}")
        print(f"  hist_all维度: {tuple(first['hist_all'].shape)}")
        if 'frame_ids_hist' in first:
            print(f"  frame_ids_hist: {first['frame_ids_hist'][:3].tolist()}... (hist_len={len(first['frame_ids_hist'])})")
            print(f"  frame_ids_fut: {first['frame_ids_fut'][:3].tolist()}... (fut_len={len(first['frame_ids_fut'])})")
    print("=" * 70 + "\n")


def save_data(data: Dict[str, list], output_dir: str, prefix: str,
              fmt: str = 'pt', skip: bool = True) -> Dict[str, str]:
    saved = {}
    for split_name, samples in data.items():
        ext = 'pt' if fmt == 'pt' else 'npy'
        out_path = os.path.join(output_dir, f'{prefix}_{split_name}.{ext}')

        if skip and os.path.exists(out_path):
            logger.info(f"已存在，跳过: {out_path}")
            saved[split_name] = out_path
            continue

        if not samples:
            logger.warning(f"{split_name} 无数据")
            continue

        pt_samples = []
        for s in samples:
            pt_s = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in s.items()}
            pt_samples.append(pt_s)

        if fmt == 'pt':
            torch.save(pt_samples, out_path)
        else:
            np.save(out_path, pt_samples)

        logger.info(f"保存 {split_name}: {out_path} ({len(pt_samples)} 样本)")
        saved[split_name] = out_path

    return saved


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    logger.info("=" * 60)
    logger.info("T2FPV 预处理开始")
    logger.info("=" * 60)
    logger.info(f"hist_len={args.hist_len}, fut_len={args.fut_len}, "
                f"frame_skip={args.frame_skip}, stride={args.stride}, "
                f"min_agents={args.min_agents}, seed={args.seed}")

    paths = resolve_paths(args)
    logger.info(f"数据: {paths['data_root']}")
    logger.info(f"输出: {paths['output']}")

    data = process_data(args, paths)
    print_statistics(data)

    saved = save_data(data, paths['output'], args.output_prefix,
                      fmt=args.save_format, skip=args.skip_if_exists)

    elapsed = time.time() - start
    logger.info(f"完成! 耗时: {elapsed:.2f}s")
    logger.info(f"输出: {list(saved.values())}")


if __name__ == '__main__':
    main()