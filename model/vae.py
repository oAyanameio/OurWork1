"""
LSTM-based Variational Autoencoder (LSTMVAE)

This module implements an LSTM-based Variational Autoencoder for sequence modeling.
The LSTMVAE combines the power of LSTM for sequential data modeling with 
Variational Autoencoder for unsupervised feature learning and generation.

Key components:
1. Encoder: LSTM-based encoder that maps input sequences to latent space
2. Decoder: LSTM-based decoder that reconstructs input from latent space
3. Reparameterization trick: Enables backpropagation through stochastic sampling

This module is used in PTINet for encoding sequential features including:
- Position trajectories
- Speed sequences  
- Pedestrian behavior sequences
- Scene attribute sequences
- Image features (when using ResNet encoder)
"""

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    LSTM编码器类
    
    将输入序列编码为隐藏状态表示。
    
    Args:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层维度
        num_layers: LSTM层数（默认1层）
    
    Input:
        x: 输入序列, shape=(batch_size, seq_length, input_size)
    
    Output:
        tuple: (hidden, cell) - LSTM的最终隐藏状态和细胞状态
               每个的shape=(num_layers, batch_size, hidden_size)
    """
    
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层定义
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,      # 输入格式为(batch, seq, feature)
            bidirectional=False,   # 单向LSTM
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入序列, shape=(batch_size, seq_length, input_size)
        
        Returns:
            tuple: (hidden, cell) - LSTM最终状态
        """
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(nn.Module):
    """
    LSTM解码器类
    
    将隐变量解码回原始输入空间，用于重构输入序列。
    
    Args:
        input_size: 输入特征维度（隐变量维度）
        hidden_size: LSTM隐藏层维度
        output_size: 输出特征维度（与编码器输入维度一致）
        num_layers: LSTM层数（默认1层）
    
    Input:
        x: 隐变量序列, shape=(batch_size, seq_length, latent_size)
        hidden: 初始隐藏状态, tuple=(hidden, cell)
    
    Output:
        tuple: (prediction, (hidden, cell))
               prediction: 重构输出, shape=(batch_size, seq_length, output_size)
    """
    
    def __init__(
        self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=1
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM层定义
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        # 全连接层将隐藏状态映射回输出维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        前向传播
        
        Args:
            x: 隐变量序列, shape=(batch_size, seq_length, latent_size)
            hidden: 初始隐藏状态, tuple=(hidden, cell)
        
        Returns:
            tuple: (prediction, (hidden, cell))
        """
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)  # 映射到输出维度
        return prediction, (hidden, cell)


class LSTMVAE(nn.Module):
    """
    LSTM-based Variational Autoencoder (LSTMVAE)
    
    核心架构:
    1. 编码器: 将输入序列编码为隐藏状态
    2. 重参数化: 从隐藏状态学习隐变量的均值和方差，进行随机采样
    3. 解码器: 将隐变量解码回输入空间
    
    损失函数:
        L = Reconstruction_Loss + KLD_weight * KL_Divergence
        
    Args:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层维度
        latent_size: 隐变量维度
        device: 计算设备 ('cuda' 或 'cpu')
    
    Input:
        x: 输入序列, shape=(batch_size, seq_length, input_size)
    
    Output:
        tuple: (loss, x_hat, z, hidden, (recon_loss, kld_loss))
               loss: 总损失
               x_hat: 重构输出
               z: 隐变量
               hidden: 编码器隐藏状态
               recon_loss: 重构损失
               kld_loss: KL散度损失
    """

    def __init__(
        self, input_size, hidden_size, latent_size, device=torch.device("cuda")
    ):
        super(LSTMVAE, self).__init__()
        self.device = device

        # 维度配置
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 1

        # 编码器和解码器
        self.lstm_enc = Encoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        # 全连接层用于学习隐变量的均值和方差
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)  # 均值
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)  # 方差的对数
        
        # 全连接层用于将隐变量映射回隐藏层维度
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)

    def reparametize(self, mu, logvar):
        """
        重参数化技巧
        
        通过重参数化，将随机采样过程转化为可微分的操作，
        使得梯度可以通过采样步骤反向传播。
        
        Args:
            mu: 隐变量均值, shape=(batch_size, latent_size)
            logvar: 隐变量方差的对数, shape=(batch_size, latent_size)
        
        Returns:
            z: 采样得到的隐变量, shape=(batch_size, latent_size)
        """
        std = torch.exp(0.5 * logvar)  # 标准差 = exp(0.5 * log(var))
        noise = torch.randn_like(std).to(self.device)  # 标准正态分布噪声
        z = mu + noise * std  # 重参数化采样
        return z

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入序列, shape=(batch_size, seq_length, input_size)
        
        Returns:
            tuple: (loss, x_hat, z, hidden, (recon_loss, kld_loss))
        """
        batch_size, seq_len, feature_dim = x.shape

        # ========== 编码阶段: 输入空间 -> 隐藏空间 ==========
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0].view(batch_size, self.hidden_size).to(self.device)

        # ========== 隐变量学习: 隐藏空间 -> 隐空间 ==========
        mean = self.fc21(enc_h)           # 学习均值
        logvar = self.fc22(enc_h)         # 学习方差对数
        z = self.reparametize(mean, logvar)  # 采样隐变量

        # ========== 解码阶段: 隐空间 -> 输入空间 ==========
        # 将隐变量扩展为序列形式（每个时间步使用相同的隐变量）
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
        
        # 解码重构
        reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)
        x_hat = reconstruct_output

        # ========== 计算损失 ==========
        losses = self.loss_function(x_hat, x, mean, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, x_hat, z, enc_hidden, (recon_loss, kld_loss)

    def loss_function(self, *args, **kwargs) -> dict:
        """
        计算VAE损失函数
        
        VAE损失由两部分组成：
        1. 重构损失 (Reconstruction Loss): 衡量输入与重构输出的差异
        2. KL散度 (KL Divergence): 衡量隐变量分布与标准正态分布的差异
        
        KL(N(mu, sigma), N(0, 1)) = log(1/sigma) + (sigma^2 + mu^2)/2 - 1/2
        
        Args:
            args: 位置参数，包含:
                  args[0]: recons - 重构输出
                  args[1]: input - 原始输入
                  args[2]: mu - 隐变量均值
                  args[3]: log_var - 隐变量方差对数
        
        Returns:
            dict: 包含损失分量的字典
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # KL散度权重（考虑批次大小）
        kld_weight = 0.00025
        
        # 重构损失：MSE损失
        recons_loss = F.mse_loss(recons, input)

        # KL散度损失
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        # 总损失
        loss = recons_loss + kld_weight * kld_loss
        
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),  # 分离不参与梯度计算
            "KLD": -kld_loss.detach(),                    # 取负使其为正值便于观察
        }