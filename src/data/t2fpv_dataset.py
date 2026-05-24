"""
T2FPV 数据集 PyTorch Dataset 类

加载预处理后的 T2FPV .pt 文件，为 OurWork1/PTINet 模型提供标准训练接口。

核心设计:
    - 每个预处理的场景样本包含 N 个agent（一个场景中多个行人）
    - Dataset 将每个场景展平为 N 个独立样本（每个agent一个样本）
    - 每个独立样本中，该agent为"当前agent"，hist_all 包含场景中所有agent
    - seq_start_end 标识场景边界，供 CoFE 模块计算 ego 相对距离
    - intent_feature（可选）由 scripts/extract_intent.py 离线写入，供 CoFE 语义锚点
"""

import os
import torch
from torch.utils.data import Dataset


class T2FPV(Dataset):
    def __init__(self,
                 data_dir: str,
                 out_dir: str,
                 dtype: str,
                 input: int,
                 output: int,
                 stride: int,
                 skip: int = 1,
                 task: str = 'bounding_box',
                 from_file: bool = False,
                 save: bool = True,
                 use_images: bool = False,
                 use_attribute: bool = False,
                 use_opticalflow: bool = False,
                 image_resize: list = None,
                 data_root: str = None,
                 **kwargs):
        print('*' * 30)
        print(f'Loading T2FPV {dtype} data ...')

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.input = input
        self.output = output
        self.stride = stride
        self.skip = skip
        self.dtype = dtype
        self.task = task
        self.use_image = use_images
        self.use_attribute = use_attribute
        self.use_opticalflow = use_opticalflow
        self.image_resize = image_resize or [240, 426]
        self.data_root = data_root or os.path.join(os.path.dirname(__file__), '..', '..', 'data')

        self.preprocessed_dir = os.path.join(self.data_root, 'processed')
        self.filename = f't2fpv_{dtype}.pt'

        loaded_data = self._load_preprocessed()
        self.data, self.agent_indices = self._flatten_samples(loaded_data)
        self.__class__.__name__ = 'T2FPV'

        self.max_agents = max((s['pos'].shape[0] for s in self.data), default=1)

        print(f'T2FPV {dtype} set loaded, {len(self.data)} scenes -> '
              f'{len(self.agent_indices)} agent-samples'
              f' (max_agents={self.max_agents})')

    def _load_preprocessed(self):
        filepath = os.path.join(self.preprocessed_dir, self.filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f'预处理数据文件不存在: {filepath}\n'
                f'请先运行预处理脚本: python scripts/preprocess.py'
            )
        # weights_only=False：预处理 .pt 含 numpy 数组，非纯权重文件
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            samples = []
            for scene_idx in range(len(data.get('pos', []))):
                pos = data['pos'][scene_idx]
                speed = data['speed'][scene_idx]
                future_pos = data['future_pos'][scene_idx]
                future_speed = data['future_speed'][scene_idx]
                ped_behavior = data['ped_behavior'][scene_idx]
                hist_all = data['hist_all'][scene_idx]
                seq_start_end = data['seq_start_end'][scene_idx]
                hist_yaw = data['hist_yaw'][scene_idx]
                samples.append({
                    'pos': pos,
                    'speed': speed,
                    'future_pos': future_pos,
                    'future_speed': future_speed,
                    'ped_behavior': ped_behavior,
                    'hist_all': hist_all,
                    'seq_start_end': seq_start_end,
                    'hist_yaw': hist_yaw,
                    'frame_ids_hist': data.get('frame_ids_hist', [None])[scene_idx] if 'frame_ids_hist' in data else None,
                    'frame_ids_fut': data.get('frame_ids_fut', [None])[scene_idx] if 'frame_ids_fut' in data else None,
                    'scene_name': data.get('scene_name', [None])[scene_idx] if 'scene_name' in data else None,
                    'agent_ids': data.get('agent_ids', [None])[scene_idx] if 'agent_ids' in data else None,
                    'intent_feature': data.get('intent_feature', [None])[scene_idx] if 'intent_feature' in data else None,
                    'intent_text': data.get('intent_text', [None])[scene_idx] if 'intent_text' in data else None,
                })
            return samples
        return data

    def _slice_time(self, tensor, dim_time=1):
        """
        按 input/skip 截取时间维，与 pos/speed 保持一致。

        hist_all 形状为 (N_agents, T_full, F)，时间在 dim=1。
        hist_yaw  形状为 (N_agents, T_full)，时间在 dim=1。
        """
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        sl = slice(None, self.input, self.skip)
        if tensor.dim() == 3:
            # (N, T, F)
            out = tensor[:, sl, :]
            if out.shape[1] < self.input:
                out = tensor[:, :self.input, :]
            return out
        if tensor.dim() == 2:
            out = tensor[:, sl]
            if out.shape[1] < self.input:
                out = tensor[:, :self.input]
            return out
        return tensor

    def _flatten_samples(self, data):
        flattened = []
        agent_indices = []
        for scene_idx, scene in enumerate(data):
            n_agents = scene['pos'].shape[0]
            for agent_idx in range(n_agents):
                flattened.append(scene)
                agent_indices.append((scene_idx, agent_idx))
        return flattened, agent_indices

    def __len__(self):
        return len(self.agent_indices)

    def __getitem__(self, index):
        scene = self.data[index]
        scene_idx, agent_idx = self.agent_indices[index]
        outputs = {}

        n_agents = scene['pos'].shape[0]
        t_in = scene['pos'].shape[1]
        t_out = scene['future_pos'].shape[1]

        ego_pos = scene['pos'][agent_idx, :self.input:self.skip]
        ego_speed = scene['speed'][agent_idx, :self.input:self.skip]
        ego_future = scene['future_pos'][agent_idx, :self.output:self.skip]
        ego_future_speed = scene['future_speed'][agent_idx, :self.output:self.skip]

        if ego_pos.shape[0] < self.input:
            ego_pos = scene['pos'][agent_idx, :self.input]
            ego_speed = scene['speed'][agent_idx, :self.input]
        if ego_future.shape[0] < self.output:
            ego_future = scene['future_pos'][agent_idx, :self.output]
            ego_future_speed = scene['future_speed'][agent_idx, :self.output]

        outputs['pos'] = ego_pos
        outputs['speed'] = ego_speed
        outputs['future_pos'] = ego_future
        outputs['future_speed'] = ego_future_speed

        outputs['ped_behavior'] = scene['ped_behavior'][agent_idx, :self.input]
        if outputs['ped_behavior'].shape[0] < self.input:
            outputs['ped_behavior'] = scene['ped_behavior'][agent_idx]

        outputs['ped_attribute'] = torch.zeros(3, dtype=torch.float32)
        outputs['scene_attribute'] = torch.zeros(10, dtype=torch.float32)

        outputs['image'] = torch.zeros(3, self.image_resize[0], self.image_resize[1])
        outputs['optical'] = torch.zeros(4, self.image_resize[0], self.image_resize[1])

        hist_all = scene.get('hist_all', None)
        seq_start_end = scene.get('seq_start_end', None)
        hist_yaw = scene['hist_yaw'] if 'hist_yaw' in scene else None

        # 与 pos 相同的时间窗口，避免 CoFE 中 hist_all(T=8) 与 hist_yaw(T=5) 不一致
        if hist_all is not None:
            hist_all = self._slice_time(hist_all, dim_time=1)
        if hist_yaw is not None:
            hist_yaw = self._slice_time(hist_yaw, dim_time=1)

        if hist_all is not None:
            n_actual = hist_all.shape[0]
            if n_actual < self.max_agents:
                pad_n = self.max_agents - n_actual
                hist_all = torch.nn.functional.pad(hist_all, (0, 0, 0, 0, 0, pad_n))
            outputs['hist_all'] = hist_all
        if seq_start_end is not None:
            outputs['seq_start_end'] = seq_start_end[0]
        if hist_yaw is not None:
            n_actual_yaw = hist_yaw.shape[0]
            if n_actual_yaw < self.max_agents:
                pad_n = self.max_agents - n_actual_yaw
                hist_yaw = torch.nn.functional.pad(hist_yaw, (0, 0, 0, pad_n))
            outputs['hist_yaw'] = hist_yaw

        outputs['scene_idx'] = scene_idx
        outputs['agent_idx'] = agent_idx
        outputs['ego_idx'] = torch.tensor(agent_idx, dtype=torch.long)

        if 'frame_ids_hist' in scene:
            outputs['frame_ids_hist'] = scene['frame_ids_hist']
        if 'frame_ids_fut' in scene:
            outputs['frame_ids_fut'] = scene['frame_ids_fut']

        # VLM 离线意图特征：形状 (N_agents, intent_dim)，由 extract_intent.py 写入
        intent_feature = scene.get('intent_feature', None)
        if intent_feature is not None:
            if not isinstance(intent_feature, torch.Tensor):
                intent_feature = torch.as_tensor(intent_feature, dtype=torch.float32)
            n_actual = intent_feature.shape[0]
            if n_actual < self.max_agents:
                pad_n = self.max_agents - n_actual
                intent_feature = torch.nn.functional.pad(intent_feature, (0, 0, 0, pad_n))
            outputs['intent_feature'] = intent_feature

        return outputs

    def collate_fn(self, batch):
        batch_size = len(batch)
        out = {}
        has_hist_all = 'hist_all' in batch[0]

        single_per_sample = ['pos', 'speed', 'future_pos', 'future_speed',
                             'ped_behavior', 'frame_ids_hist', 'frame_ids_fut']
        for key in single_per_sample:
            if key in batch[0]:
                out[key] = torch.stack([b[key] for b in batch], dim=0)

        single_per_batch = ['ped_attribute', 'scene_attribute', 'image', 'optical']
        for key in single_per_batch:
            if key in batch[0]:
                out[key] = torch.stack([b[key] for b in batch], dim=0)

        if has_hist_all:
            hist_all_batch = []
            seq_start_end_list = []
            hist_yaw_list = []
            ego_idx_list = []
            offset = 0
            for b in batch:
                ha = b['hist_all']
                sse = b['seq_start_end'].clone()
                n_actual = int(sse[1] - sse[0])
                sse[0] += offset
                sse[1] = sse[0] + n_actual
                ego_global = sse[0].item() + b['ego_idx'].item()
                offset += n_actual
                hist_all_batch.append(ha)
                seq_start_end_list.append(sse)
                ego_idx_list.append(ego_global)
                if 'hist_yaw' in b:
                    hist_yaw_list.append(b['hist_yaw'])

            out['hist_all'] = torch.stack(hist_all_batch, dim=0).permute(0, 2, 1, 3).contiguous()
            out['seq_start_end'] = torch.stack(seq_start_end_list, dim=0)
            out['ego_idx'] = torch.tensor(ego_idx_list, dtype=torch.long)
            if hist_yaw_list:
                out['hist_yaw'] = torch.stack(hist_yaw_list, dim=0).permute(0, 2, 1).contiguous()

            # 场景级意图特征 (B, N_max, D) → PTINet 展平为 (B*N, D) 供 CoFE 使用
            if 'intent_feature' in batch[0]:
                intent_batch = []
                for b in batch:
                    intent_batch.append(b['intent_feature'])
                out['intent_feature'] = torch.stack(intent_batch, dim=0)

        return out