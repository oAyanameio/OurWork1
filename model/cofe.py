import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CoFE(nn.Module):
    """
    CoFE: Correction Feature Embedding（轨迹去噪修复模块）

    FPV（第一人称视角）完整版本，匹配T2FPV原始实现。

    核心能力:
      - 偏航角编码：4步变换（ego相对化 → 归一化 → 弧度化 → cos/sin编码）
      - Ego相对距离：通过seq_start_end计算相对于自车的坐标
      - ResNet视觉融合：可选，融合2048维ResNet特征
      - 帧间位移修正：在位移空间中做GRU编解码
      - 绝对坐标恢复：cumsum + initial_pos

    特征空间（8维，no_abs=True）:
      dims 0-1: xy - xy[0]        相对首帧位移
      dims 2-3: offset_xy          ego相对坐标
      dims 4-5: cos/sin(yaw)      偏航角编码
      dims 6-7: rel                帧间位移（速度）
      通过 idxs 参数选择子集送入GRU（默认[6,7]只选位移）

    数据流:
      绝对坐标 → 特征构建 → idxs选择 → GRU编码 → GRU解码 → 位移还原 → 修正后绝对坐标
    """

    def __init__(self, input_size=2, hidden_size=96, num_layers=2,
                 use_resnet=False, no_abs=True, idxs=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.no_abs = no_abs
        self.use_resnet = use_resnet
        self.idxs = idxs if idxs is not None else [6, 7]
        self.offset_idxs = torch.tensor(self.idxs, dtype=torch.long)
        self.feat_dim = len(self.idxs)

        self.f_offset = MLP(self.feat_dim, [hidden_size], hidden_size)

        if use_resnet:
            self.f_resnet = MLP(2048, [hidden_size], hidden_size)
            corr_enc_in = hidden_size * 3
        else:
            self.f_resnet = None
            corr_enc_in = hidden_size * 2

        self.corr_enc = MLP(corr_enc_in, [hidden_size, 64], hidden_size)
        self.corr_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec = MLP(hidden_size, [64, hidden_size], self.feat_dim)

        self.criterion = nn.MSELoss()

    def to(self, *args, **kwargs):
        self.offset_idxs = self.offset_idxs.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def ego_dists(hist_abs, seq_start_end):
        """
        向量化计算每个agent相对于其场景ego的坐标偏移
        
        Args:
            hist_abs: (T, N, 2) 世界坐标
            seq_start_end: (S, 2) 场景边界索引
        
        Returns:
            hist_ego_abs: (T, N, 2) ego相对坐标
        """
        T, N, _ = hist_abs.shape
        device = hist_abs.device
        
        scene_idx = torch.zeros(N, dtype=torch.long, device=device)
        for i, (start, end) in enumerate(seq_start_end):
            scene_idx[start:end] = i
        
        ego_indices = seq_start_end[:, 0]
        
        ego_coords = hist_abs[:, ego_indices, :]
        
        indices = scene_idx.view(1, N, 1).expand(T, N, 2)
        result = ego_coords.gather(dim=1, index=indices)
        
        return hist_abs - result

    @staticmethod
    def encode_yaw(hist_yaw, seq_start_end):
        """
        向量化计算偏航角的ego相对编码
        
        Args:
            hist_yaw: (T, N) 原始偏航角
            seq_start_end: (S, 2) 场景边界索引
        
        Returns:
            yaw_enc: (T, N, 2) cos/sin编码的偏航角
        """
        T, N = hist_yaw.shape
        device = hist_yaw.device
        
        scene_idx = torch.zeros(N, dtype=torch.long, device=device)
        for i, (start, end) in enumerate(seq_start_end):
            scene_idx[start:end] = i
        
        ego_indices = seq_start_end[:, 0]
        
        ego_yaw = hist_yaw[:, ego_indices]
        
        indices = scene_idx.view(1, N).expand(T, N)
        result = ego_yaw.gather(dim=1, index=indices)
        
        offset_yaw_rel = hist_yaw - result
        
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)

    def build_features(self, hist_abs_pred, hist_yaw_pred, hist_seq_start_end):
        xy_pred = hist_abs_pred
        rel_pred = torch.zeros_like(xy_pred)
        rel_pred[1:] = xy_pred[1:] - xy_pred[:-1]
        offset_xy_pred = self.ego_dists(xy_pred, hist_seq_start_end)
        offset_yaw_pred = self.encode_yaw(hist_yaw_pred, hist_seq_start_end)

        if self.no_abs:
            offset_pred = torch.cat([
                xy_pred - xy_pred[0],
                offset_xy_pred,
                offset_yaw_pred,
                rel_pred
            ], dim=-1)
        else:
            offset_pred = torch.cat([
                xy_pred,
                offset_xy_pred,
                offset_yaw_pred,
                rel_pred
            ], dim=-1)

        return offset_pred[..., self.offset_idxs]

    def _encode_resnet_step(self, hist_resnet, t, num_agents, device):
        """编码单帧 ResNet 特征，并处理特征缺失的机器人部署场景。

        cofe_use_resnet=True 时 corr_enc 的输入维度会包含视觉分支。若某个实验
        暂时没有预提取 ResNet 特征，直接拼空张量会造成维度不匹配；这里用
        2048 维零特征兜底，保证模型结构和参数维度保持一致。
        """
        if self.f_resnet is None:
            return torch.empty((num_agents, 0), device=device)
        if hist_resnet is None:
            hist_resnet_t = torch.zeros(num_agents, 2048, device=device)
        else:
            # 正常路径：使用 T2FPV/机器人数据集中随时间对齐的视觉特征。
            hist_resnet_t = hist_resnet[t]
        return self.f_resnet(hist_resnet_t)

    def train_correction(self, hist_abs_gt, hist_yaw_gt, hist_abs_pred,
                         hist_yaw_pred, hist_resnet, hist_seq_start_end):
        timesteps, num_agents, _ = hist_abs_gt.shape
        device = hist_abs_gt.device

        MSE = torch.zeros(1).to(device)
        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)

        # 训练 CoFE 时，GT 和噪声预测轨迹必须进入相同特征空间，
        # 这样修正损失监督的是位移/yaw/ego-relative 等被 idxs 选中的特征。
        offset_gt = self.build_features(hist_abs_gt, hist_yaw_gt, hist_seq_start_end)
        offset_pred = self.build_features(hist_abs_pred, hist_yaw_pred, hist_seq_start_end)

        for t in range(timesteps):
            f_offset_t = self.f_offset(offset_pred[t])
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)
            x_enc = torch.cat([f_offset_t, f_resnet_t, h[-1]], dim=-1)
            x_corr = self.corr_enc(x_enc)
            _, h = self.corr_rnn(x_corr.unsqueeze(0), h)

        for t in range(timesteps):
            x_dec = self.corr_dec(h[-1])
            MSE += torch.sqrt(self.criterion(x_dec, offset_gt[t]))
            x_dec_feat = self.f_offset(x_dec)
            _, h = self.corr_dec_rnn(x_dec_feat.unsqueeze(0), h)

        return MSE

    def infer_correction(self, hist_abs_pred, hist_yaw_pred=None,
                         hist_resnet=None, hist_seq_start_end=None):
        timesteps, num_agents, _ = hist_abs_pred.shape
        device = hist_abs_pred.device

        rel_pred = torch.zeros_like(hist_abs_pred)
        rel_pred[1:] = hist_abs_pred[1:] - hist_abs_pred[:-1]

        if hist_yaw_pred is not None and hist_seq_start_end is not None:
            # FPV/T2FPV 路径：使用 ego 相对坐标、yaw 编码和相对位移构造 CoFE 特征。
            offset_pred = self.build_features(
                hist_abs_pred, hist_yaw_pred, hist_seq_start_end
            )
        else:
            # BEV/旧 PTINet 路径：没有 yaw/场景边界时退化为纯相对位移修正。
            offset_pred = rel_pred

        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)

        for t in range(timesteps):
            f_offset_t = self.f_offset(offset_pred[t])
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)
            x_enc = torch.cat([f_offset_t, f_resnet_t, h[-1]], dim=-1)
            x_corr = self.corr_enc(x_enc)
            _, h = self.corr_rnn(x_corr.unsqueeze(0), h)

        outputs = []
        for t in range(timesteps):
            dec_out = self.corr_dec(h[-1])
            outputs.append(dec_out.unsqueeze(0))
            dec_feat = self.f_offset(dec_out)
            _, h = self.corr_dec_rnn(dec_feat.unsqueeze(0), h)

        samples = torch.cat(outputs, dim=0)

        if self.no_abs and hist_yaw_pred is not None and hist_seq_start_end is not None:
            samples = torch.cumsum(samples[..., -2:], dim=0) + hist_abs_pred[0:1]
        else:
            samples = torch.cumsum(samples, dim=0) + hist_abs_pred[0:1]

        return samples

    def forward(self, x):
        pos_seq_first = x.permute(1, 0, 2)
        corrected_seq_first = self.infer_correction(pos_seq_first)
        return corrected_seq_first.permute(1, 0, 2)
