import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron）
    
    一个简单的全连接神经网络，由多个线性层和激活函数组成。
    常用于特征变换、维度映射等场景。
    
    Args:
        in_size: 输入特征维度
        hidden_sizes: 隐藏层维度列表，如 [128, 64] 表示两个隐藏层
        out_size: 输出特征维度
        dropout: Dropout概率，用于防止过拟合，默认为0（不使用）
    """
    def __init__(self, in_size, hidden_sizes, out_size, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_size
        # 遍历隐藏层配置，逐层构建网络
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))      # 线性变换
            layers.append(nn.LeakyReLU())          # 激活函数（带泄露的ReLU）
            if dropout > 0:
                layers.append(nn.Dropout(dropout)) # Dropout层（可选）
            prev = h
        layers.append(nn.Linear(prev, out_size))   # 输出层
        self.net = nn.Sequential(*layers)         # 组合成序列模型

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_size)
        
        Returns:
            输出张量，形状为 (batch_size, out_size)
        """
        return self.net(x)


class CoFE(nn.Module):
    """
    CoFE: Correction Feature Embedding（轨迹去噪修复模块）
    
    这是一个用于轨迹预测任务的轨迹修正模块，主要功能是对有噪声的预测轨迹进行去噪和修复。
    
    核心设计理念：
    - 将原始轨迹转换到特征空间（包含位移、ego相对坐标、偏航角等）
    - 使用GRU编码器-解码器架构在特征空间中进行修正
    - 最后将修正后的特征还原为绝对坐标轨迹
    
    支持两种模式：
    1. FPV（第一人称视角）模式：使用完整的特征空间（包含yaw和场景边界信息）
    2. BEV（鸟瞰视角）模式：仅使用相对位移进行修正
    
    特征空间设计（8维，当no_abs=True时）:
      dims 0-1: xy - xy[0]        相对首帧的位移
      dims 2-3: offset_xy          相对于自车(ego)的坐标偏移
      dims 4-5: cos/sin(yaw)      偏航角的余弦/正弦编码
      dims 6-7: rel                帧间位移（速度信息）
    
    模型架构：
      - f_offset: 将选中的特征维度映射到隐藏层维度
      - f_resnet: 可选的视觉特征处理分支（ResNet-2048维）
      - corr_enc: 编码器特征融合模块
      - corr_rnn: GRU编码器（读取输入序列，学习隐藏状态）
      - corr_dec_rnn: GRU解码器（基于隐藏状态生成修正序列）
      - corr_dec: 解码器输出层（将隐藏状态映射回特征空间）
    """

    def __init__(self, input_size=2, hidden_size=96, num_layers=2,
                 use_resnet=False, no_abs=True, idxs=None):
        """
        初始化CoFE模块
        
        Args:
            input_size: 输入坐标维度，通常为2（x, y）
            hidden_size: GRU和MLP的隐藏层维度
            num_layers: GRU的层数
            use_resnet: 是否使用ResNet视觉特征
            no_abs: 是否使用相对坐标而非绝对坐标
            idxs: 从8维特征空间中选择哪些维度送入GRU，默认[6,7]只选择位移特征
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.no_abs = no_abs                    # 是否使用相对坐标
        self.use_resnet = use_resnet            # 是否使用视觉特征
        self.idxs = idxs if idxs is not None else [6, 7]  # 默认只处理位移特征
        self.offset_idxs = torch.tensor(self.idxs, dtype=torch.long)  # 特征选择索引
        self.feat_dim = len(self.idxs)          # 实际送入GRU的特征维度

        # 特征映射网络：将选中的特征维度映射到隐藏层
        self.f_offset = MLP(self.feat_dim, [hidden_size], hidden_size)

        # 视觉特征处理分支（可选）
        if use_resnet:
            self.f_resnet = MLP(2048, [hidden_size], hidden_size)  # ResNet 2048维 -> hidden_size
            corr_enc_in = hidden_size * 3  # 融合offset + resnet + hidden_state
        else:
            self.f_resnet = None
            corr_enc_in = hidden_size * 2  # 融合offset + hidden_state

        # 编码器特征融合：将多种输入特征融合为GRU输入
        self.corr_enc = MLP(corr_enc_in, [hidden_size, 64], hidden_size)
        
        # GRU编码器：处理输入序列，学习隐藏状态
        self.corr_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        
        # GRU解码器：基于编码器输出的隐藏状态生成修正序列
        self.corr_dec_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        
        # 解码器输出层：将GRU隐藏状态映射回特征维度
        self.corr_dec = MLP(hidden_size, [64, hidden_size], self.feat_dim)

        # 损失函数：用于训练时计算预测误差
        self.criterion = nn.MSELoss()

    def to(self, *args, **kwargs):
        """
        将模型移动到指定设备（CPU/GPU）
        
        重写此方法以确保offset_idxs也被正确移动到目标设备
        """
        self.offset_idxs = self.offset_idxs.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def ego_dists(hist_abs, seq_start_end):
        """
        向量化计算每个智能体(agent)相对于其场景中自车(ego)的坐标偏移
        
        场景结构：一个批次中可能包含多个场景，每个场景有一个ego车辆和多个其他agent
        seq_start_end定义了每个场景的agent索引范围
        
        Args:
            hist_abs: (T, N, 2) 世界坐标系下的绝对坐标，T=时间步，N=agent数量
            seq_start_end: (S, 2) 场景边界索引，S=场景数量，每行[start, end]表示一个场景
            
        Returns:
            hist_ego_abs: (T, N, 2) ego相对坐标，即每个agent相对于其场景ego的坐标
        """
        T, N, _ = hist_abs.shape
        device = hist_abs.device
        
        # 为每个agent分配场景索引
        scene_idx = torch.zeros(N, dtype=torch.long, device=device)
        for i, (start, end) in enumerate(seq_start_end):
            scene_idx[start:end] = i  # agent从start到end属于第i个场景
        
        # 获取每个场景的ego索引（每个场景的第一个agent是ego）
        ego_indices = seq_start_end[:, 0]
        
        # 获取每个场景ego的坐标：(T, S, 2)
        ego_coords = hist_abs[:, ego_indices, :]
        
        # 构建索引，用于从ego_coords中获取每个agent对应的ego坐标
        indices = scene_idx.view(1, N, 1).expand(T, N, 2)
        result = ego_coords.gather(dim=1, index=indices)
        
        # 计算相对坐标：agent坐标 - 对应场景的ego坐标
        return hist_abs - result

    @staticmethod
    def encode_yaw(hist_yaw, seq_start_end):
        """
        向量化计算偏航角的ego相对编码
        
        偏航角编码步骤：
        1. ego相对化：计算每个agent相对于场景ego的偏航角差
        2. 归一化：将角度归一化到[-180, 180]范围
        3. 弧度化：转换为弧度
        4. cos/sin编码：使用余弦和正弦函数将角度编码为二维向量
        
        Args:
            hist_yaw: (T, N) 原始偏航角（度数）
            seq_start_end: (S, 2) 场景边界索引
            
        Returns:
            yaw_enc: (T, N, 2) cos/sin编码的偏航角特征
        """
        T, N = hist_yaw.shape
        device = hist_yaw.device
        
        # 为每个agent分配场景索引（与ego_dists相同逻辑）
        scene_idx = torch.zeros(N, dtype=torch.long, device=device)
        for i, (start, end) in enumerate(seq_start_end):
            scene_idx[start:end] = i
        
        # 获取每个场景ego的偏航角
        ego_indices = seq_start_end[:, 0]
        ego_yaw = hist_yaw[:, ego_indices]
        
        # 获取每个agent对应的ego偏航角
        indices = scene_idx.view(1, N).expand(T, N)
        result = ego_yaw.gather(dim=1, index=indices)
        
        # 步骤1：计算相对于ego的偏航角差
        offset_yaw_rel = hist_yaw - result
        
        # 步骤2：归一化到[-180, 180]
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        
        # 步骤3：转换为弧度
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        
        # 步骤4：cos/sin编码（周期性特征的标准编码方式）
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)

    def build_features(self, hist_abs_pred, hist_yaw_pred, hist_seq_start_end):
        """
        构建CoFE的输入特征空间
        
        根据配置将原始轨迹转换为8维特征空间，然后根据idxs选择子集。
        
        Args:
            hist_abs_pred: (T, N, 2) 预测的绝对坐标轨迹
            hist_yaw_pred: (T, N) 预测的偏航角
            hist_seq_start_end: (S, 2) 场景边界索引
            
        Returns:
            (T, N, feat_dim) 选中的特征子集，feat_dim = len(idxs)
        """
        xy_pred = hist_abs_pred
        
        # 计算帧间位移（速度）：rel[t] = xy[t] - xy[t-1]
        rel_pred = torch.zeros_like(xy_pred)
        rel_pred[1:] = xy_pred[1:] - xy_pred[:-1]
        
        # 计算ego相对坐标
        offset_xy_pred = self.ego_dists(xy_pred, hist_seq_start_end)
        
        # 计算偏航角编码
        offset_yaw_pred = self.encode_yaw(hist_yaw_pred, hist_seq_start_end)

        # 根据no_abs配置决定是否使用绝对坐标
        if self.no_abs:
            # 使用相对坐标：相对首帧位移 + ego相对坐标 + yaw编码 + 帧间位移
            offset_pred = torch.cat([
                xy_pred - xy_pred[0],    # 相对首帧的位移
                offset_xy_pred,           # ego相对坐标
                offset_yaw_pred,          # 偏航角编码
                rel_pred                  # 帧间位移
            ], dim=-1)  # 结果维度：(T, N, 8)
        else:
            # 使用绝对坐标：绝对坐标 + ego相对坐标 + yaw编码 + 帧间位移
            offset_pred = torch.cat([
                xy_pred,                  # 绝对坐标
                offset_xy_pred,           # ego相对坐标
                offset_yaw_pred,          # 偏航角编码
                rel_pred                  # 帧间位移
            ], dim=-1)  # 结果维度：(T, N, 8)

        # 根据idxs选择特征子集（默认只选择帧间位移dims 6-7）
        return offset_pred[..., self.offset_idxs]

    def _encode_resnet_step(self, hist_resnet, t, num_agents, device):
        """
        编码单帧ResNet视觉特征，并处理特征缺失的情况
        
        当配置use_resnet=True但实际没有视觉特征时，使用零向量兜底，
        保证模型结构和参数维度保持一致。
        
        Args:
            hist_resnet: (T, N, 2048) ResNet视觉特征序列，可能为None
            t: 当前时间步
            num_agents: agent数量
            device: 计算设备
            
        Returns:
            (num_agents, hidden_size) 编码后的视觉特征
        """
        # 如果没有配置ResNet分支，返回空张量
        if self.f_resnet is None:
            return torch.empty((num_agents, 0), device=device)
        
        # 如果没有提供视觉特征，使用零向量兜底
        if hist_resnet is None:
            hist_resnet_t = torch.zeros(num_agents, 2048, device=device)
        else:
            # 正常路径：获取当前时间步的视觉特征
            hist_resnet_t = hist_resnet[t]
        
        # 通过MLP将2048维特征映射到hidden_size
        return self.f_resnet(hist_resnet_t)

    def train_correction(self, hist_abs_gt, hist_yaw_gt, hist_abs_pred,
                         hist_yaw_pred, hist_resnet, hist_seq_start_end):
        """
        训练模式下的轨迹修正
        
        输入带噪声的预测轨迹和真实轨迹(GT)，通过GRU编解码学习修正能力，
        损失函数监督修正后的特征与真实特征的差异。
        
        Args:
            hist_abs_gt: (T, N, 2) 真实的绝对坐标轨迹
            hist_yaw_gt: (T, N) 真实的偏航角
            hist_abs_pred: (T, N, 2) 预测的（带噪声的）绝对坐标轨迹
            hist_yaw_pred: (T, N) 预测的偏航角
            hist_resnet: (T, N, 2048) ResNet视觉特征（可选）
            hist_seq_start_end: (S, 2) 场景边界索引
            
        Returns:
            MSE: 修正损失（所有时间步的RMSE之和）
        """
        timesteps, num_agents, _ = hist_abs_gt.shape
        device = hist_abs_gt.device

        # 初始化损失和GRU隐藏状态
        MSE = torch.zeros(1).to(device)
        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)

        # 关键：GT和预测轨迹都转换到相同的特征空间
        # 这样损失监督的是位移/yaw/ego-relative等被idxs选中的特征
        offset_gt = self.build_features(hist_abs_gt, hist_yaw_gt, hist_seq_start_end)
        offset_pred = self.build_features(hist_abs_pred, hist_yaw_pred, hist_seq_start_end)

        # ========== 编码器阶段：读取输入序列 ==========
        for t in range(timesteps):
            # 编码当前时间步的特征
            f_offset_t = self.f_offset(offset_pred[t])  # 特征映射
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)  # 视觉特征
            x_enc = torch.cat([f_offset_t, f_resnet_t, h[-1]], dim=-1)  # 融合输入
            x_corr = self.corr_enc(x_enc)  # 特征融合
            _, h = self.corr_rnn(x_corr.unsqueeze(0), h)  # GRU前向，更新隐藏状态

        # ========== 解码器阶段：生成修正序列 ==========
        for t in range(timesteps):
            x_dec = self.corr_dec(h[-1])  # 从隐藏状态解码出修正特征
            MSE += torch.sqrt(self.criterion(x_dec, offset_gt[t]))  # 计算RMSE损失
            x_dec_feat = self.f_offset(x_dec)  # 将输出再编码为特征
            _, h = self.corr_dec_rnn(x_dec_feat.unsqueeze(0), h)  # 更新解码器隐藏状态

        return MSE

    def infer_correction(self, hist_abs_pred, hist_yaw_pred=None,
                         hist_resnet=None, hist_seq_start_end=None):
        """
        推理模式下的轨迹修正
        
        根据输入的预测轨迹（可能带噪声），输出修正后的轨迹。
        
        Args:
            hist_abs_pred: (T, N, 2) 预测的绝对坐标轨迹
            hist_yaw_pred: (T, N) 预测的偏航角（FPV模式需要）
            hist_resnet: (T, N, 2048) ResNet视觉特征（可选）
            hist_seq_start_end: (S, 2) 场景边界索引（FPV模式需要）
            
        Returns:
            samples: (T, N, 2) 修正后的绝对坐标轨迹
        """
        timesteps, num_agents, _ = hist_abs_pred.shape
        device = hist_abs_pred.device

        # 计算帧间位移
        rel_pred = torch.zeros_like(hist_abs_pred)
        rel_pred[1:] = hist_abs_pred[1:] - hist_abs_pred[:-1]

        # 判断运行模式
        if hist_yaw_pred is not None and hist_seq_start_end is not None:
            # FPV/T2FPV路径：使用完整特征空间
            offset_pred = self.build_features(
                hist_abs_pred, hist_yaw_pred, hist_seq_start_end
            )
        else:
            # BEV/旧PTINet路径：退化为纯位移修正
            offset_pred = rel_pred

        # 初始化GRU隐藏状态
        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)

        # ========== 编码器阶段 ==========
        for t in range(timesteps):
            f_offset_t = self.f_offset(offset_pred[t])
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)
            x_enc = torch.cat([f_offset_t, f_resnet_t, h[-1]], dim=-1)
            x_corr = self.corr_enc(x_enc)
            _, h = self.corr_rnn(x_corr.unsqueeze(0), h)

        # ========== 解码器阶段 ==========
        outputs = []
        for t in range(timesteps):
            dec_out = self.corr_dec(h[-1])  # 解码出修正后的特征
            outputs.append(dec_out.unsqueeze(0))  # 保存输出
            dec_feat = self.f_offset(dec_out)  # 再编码
            _, h = self.corr_dec_rnn(dec_feat.unsqueeze(0), h)  # 更新状态

        # 拼接所有时间步的输出
        samples = torch.cat(outputs, dim=0)

        # 将修正后的特征还原为绝对坐标
        # 通过累积求和(cumsum)将位移转换为绝对位置
        if self.no_abs and hist_yaw_pred is not None and hist_seq_start_end is not None:
            # FPV模式：只使用最后2维（位移特征）进行累积
            samples = torch.cumsum(samples[..., -2:], dim=0) + hist_abs_pred[0:1]
        else:
            # BEV模式：使用所有维度进行累积
            samples = torch.cumsum(samples, dim=0) + hist_abs_pred[0:1]

        return samples

    def forward(self, x):
        """
        前向传播接口（简化版）
        
        输入输出维度保持一致，方便集成到其他模型中。
        
        Args:
            x: (N, T, 2) 输入轨迹，N=batch/agent维度，T=时间维度
            
        Returns:
            (N, T, 2) 修正后的轨迹
        """
        # 调整维度顺序：(N, T, 2) -> (T, N, 2) 以匹配infer_correction的输入格式
        pos_seq_first = x.permute(1, 0, 2)
        
        # 执行轨迹修正（使用BEV模式，不需要yaw和场景边界）
        corrected_seq_first = self.infer_correction(pos_seq_first)
        
        # 恢复原始维度顺序：(T, N, 2) -> (N, T, 2)
        return corrected_seq_first.permute(1, 0, 2)
