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

    基于GRU编码器-解码器架构，对第一视角下带噪声的检测轨迹进行去噪修复。
    本模块适配PTINet的BEV视角轨迹预测，简化了T2FPV原始实现中的
    yaw角度编码、ResNet视觉融合、ego相对距离等FPV特有组件，
    仅对轨迹序列本身进行编解码修复。

    架构:
        f_offset:    轨迹特征编码器MLP
        corr_enc:    融合编码器MLP（拼接编码特征和GRU隐状态）
        corr_rnn:    编码器GRU
        corr_dec_rnn:解码器GRU
        corr_dec:    输出解码器MLP

    数据流:
        1. 编码阶段：逐时间步将输入轨迹编码到隐空间
        2. 解码阶段：从隐状态逐时间步解码出修正后的轨迹
    """

    def __init__(self, input_size=4, hidden_size=96, num_layers=2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.f_offset = MLP(input_size, [hidden_size], hidden_size)
        self.corr_enc = MLP(hidden_size * 2, [hidden_size, 64], hidden_size)
        self.corr_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec = MLP(hidden_size, [64, hidden_size], input_size)

    def infer_correction(
        self,
        hist_abs_pred,
        hist_yaw_pred=None,
        hist_resnet=None,
        hist_seq_start_end=None,
    ):
        """
        CoFE推理修正接口 —— 严格遵循T2FPV原始设计风格

        编码器逐时间步读入带噪声轨迹，解码器从隐状态逐时间步生成修正轨迹。

        Args:
            hist_abs_pred:    带噪声的轨迹序列
                              形状: (timesteps, num_agents, input_size)
                              说明: 时间步在第0维，与T2FPV原始设计一致
            hist_yaw_pred:    可选，偏航角数据（PTINet中未使用，保留接口兼容）
            hist_resnet:      可选，ResNet视觉特征（PTINet中未使用，保留接口兼容）
            hist_seq_start_end: 可选，场景边界索引（PTINet中未使用，保留接口兼容）

        Returns:
            corrected: 修正后的轨迹序列
                       形状: (timesteps, num_agents, input_size)
                       与输入形状相同，仅内容被修复
        """
        timesteps, num_agents, _ = hist_abs_pred.shape
        device = hist_abs_pred.device

        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)

        for t in range(timesteps):
            x_t = hist_abs_pred[t]
            f_t = self.f_offset(x_t)
            enc_input = torch.cat([f_t, h[-1]], dim=-1)
            enc_out = self.corr_enc(enc_input)
            _, h = self.corr_rnn(enc_out.unsqueeze(0), h)

        outputs = []
        for t in range(timesteps):
            dec_out = self.corr_dec(h[-1])
            outputs.append(dec_out.unsqueeze(0))
            dec_feat = self.f_offset(dec_out)
            _, h = self.corr_dec_rnn(dec_feat.unsqueeze(0), h)

        return torch.cat(outputs, dim=0)

    def forward(self, x):
        """
        前向传播（简便接口）

        与PTINet原生的batch_first格式直接兼容，
        内部自动permute后调用infer_correction再permute回来。

        Args:
            x: 带噪声的轨迹序列
               形状: (batch_size, seq_len, input_size)

        Returns:
            corrected: 修正后的轨迹序列
                       形状: (batch_size, seq_len, input_size)
        """
        pos_seq_first = x.permute(1, 0, 2)
        corrected_seq_first = self.infer_correction(pos_seq_first)
        return corrected_seq_first.permute(1, 0, 2)