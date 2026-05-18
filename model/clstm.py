"""
Convolutional LSTM (ConvLSTM) Module

This module implements a Convolutional LSTM architecture for spatiotemporal 
sequence modeling. ConvLSTM extends standard LSTM by replacing matrix 
multiplications with convolutional operations, making it particularly suited 
for processing video sequences and other grid-structured data.

Based on implementation from: https://github.com/automan000/Convolution_LSTM_pytorch

Key features:
1. ConvLSTMCell: Single time step convolutional LSTM cell
2. ConvLSTM: Multi-layer ConvLSTM network with pooling and batch normalization

In PTINet, ConvLSTM is used to extract spatiotemporal features from image sequences,
capturing both spatial patterns (appearance) and temporal dynamics (motion).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元
    
    ConvLSTM将标准LSTM中的全连接层替换为卷积层，能够保留输入的空间结构信息。
    每个门（输入门、遗忘门、输出门）和细胞状态更新都通过卷积操作实现。
    
    Args:
        input_channels: 输入通道数（如RGB图像为3通道）
        hidden_channels: 隐藏状态通道数
        kernel_size: 卷积核大小
        conv_stride: 卷积步长
    
    Input:
        x: 当前时间步输入, shape=(batch_size, input_channels, height, width)
        h: 上一时间步隐藏状态, shape=(batch_size, hidden_channels, height, width)
        c: 上一时间步细胞状态, shape=(batch_size, hidden_channels, height, width)
    
    Output:
        tuple: (ch, cc) - 当前时间步的隐藏状态和细胞状态
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride):
        super(ConvLSTMCell, self).__init__()

        # 确保隐藏通道数为偶数（用于门控机制的对称性）
        assert hidden_channels % 2 == 0

        # 保存参数
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4  # 输入门、遗忘门、细胞更新、输出门
        self.conv_stride = conv_stride
      
        # 计算padding以保持空间维度不变
        self.padding = int((kernel_size - 1) / 2)

        # 卷积层定义 - 每个门使用独立的卷积核
        # 输入门卷积层
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
                             self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
                             self.kernel_size, 1, self.padding, bias=False)
        
        # 遗忘门卷积层
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
                             self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
                             self.kernel_size, 1, self.padding, bias=False)
        
        # 细胞状态更新卷积层
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
                             self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
                             self.kernel_size, 1, self.padding, bias=False)
        
        # 输出门卷积层
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
                             self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
                             self.kernel_size, 1, self.padding, bias=False)

        # 细胞状态门控权重（延迟初始化）
        self.Wci = None  # 输入门的细胞状态权重
        self.Wcf = None  # 遗忘门的细胞状态权重
        self.Wco = None  # 输出门的细胞状态权重

    def forward(self, x, h, c):
        """
        前向传播
        
        ConvLSTM门控机制：
        1. 输入门(i): 控制哪些信息进入细胞状态
        2. 遗忘门(f): 控制哪些信息从细胞状态中遗忘
        3. 细胞状态更新(c): 更新细胞状态
        4. 输出门(o): 控制细胞状态中哪些信息输出为隐藏状态
        
        Args:
            x: 当前时间步输入, shape=(batch_size, input_channels, H, W)
            h: 上一时间步隐藏状态, shape=(batch_size, hidden_channels, H, W)
            c: 上一时间步细胞状态, shape=(batch_size, hidden_channels, H, W)
        
        Returns:
            tuple: (ch, cc) - 当前时间步的隐藏状态和细胞状态
        """
        # 输入门: i = sigmoid(Wxi*x + Whi*h + Wci*c)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        
        # 遗忘门: f = sigmoid(Wxf*x + Whf*h + Wcf*c)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        
        # 细胞状态更新: c = f*c_prev + i*tanh(Wxc*x + Whc*h)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        
        # 输出门: o = sigmoid(Wxo*x + Who*h + Wco*c)
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        
        # 隐藏状态: h = o*tanh(c)
        ch = co * torch.tanh(cc)
        
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        """
        初始化隐藏状态和细胞状态
        
        Args:
            batch_size: 批次大小
            hidden: 隐藏通道数
            shape: 输入空间维度 (height, width)
        
        Returns:
            tuple: (h, c) - 初始隐藏状态和细胞状态
        """
        # 延迟初始化细胞状态门控权重
        if self.Wci is None:
            output_shape = (shape[0] // self.conv_stride, shape[1] // self.conv_stride)
            self.Wci = Variable(torch.zeros(1, hidden, output_shape[0], output_shape[1])).to(torch.device('cuda'))
            self.Wcf = Variable(torch.zeros(1, hidden, output_shape[0], output_shape[1])).to(torch.device('cuda'))
            self.Wco = Variable(torch.zeros(1, hidden, output_shape[0], output_shape[1])).to(torch.device('cuda'))
        else:
            # 验证输入形状是否匹配
            assert shape[0] // self.conv_stride == self.Wci.size()[2], \
                'Input Height Mismatched! %d vs %d' % (shape[0] // self.conv_stride, self.Wci.size()[2])
            assert shape[1] // self.conv_stride == self.Wci.size()[3], 'Input Width Mismatched!'
        
        # 返回初始化为零的隐藏状态和细胞状态
        output_shape = (shape[0] // self.conv_stride, shape[1] // self.conv_stride)
        return (
            Variable(torch.zeros(batch_size, hidden, output_shape[0], output_shape[1])).to(torch.device('cuda')),
            Variable(torch.zeros(batch_size, hidden, output_shape[0], output_shape[1])).to(torch.device('cuda'))
        )


class ConvLSTM(nn.Module):
    """
    多层卷积LSTM网络
    
    由多个ConvLSTMCell堆叠而成，每层之间可以应用池化和批归一化。
    支持指定有效输出时间步，用于提取特定时间步的特征。
    
    Args:
        input_channels: 输入通道数
        hidden_channels: 各层隐藏通道数列表
        kernel_size: 卷积核大小
        conv_stride: 卷积步长
        pool_kernel_size: 最大池化核大小（默认(2,2)）
        step: 时间步数（序列长度）
        effective_step: 需要记录输出的时间步索引列表
        batch_normalization: 是否使用批归一化（默认True）
        dropout: dropout率（默认0，即不使用）
    
    Input:
        input: 时空序列输入, shape=(batch_size, seq_length, input_channels, height, width)
    
    Output:
        tuple: (outputs, (x, new_c))
               outputs: 有效时间步的输出列表
               x: 最后一层最后时间步的隐藏状态
               new_c: 最后一层最后时间步的细胞状态（经过池化）
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride,
                 pool_kernel_size=(2, 2), step=1, effective_step=[1],
                 batch_normalization=True, dropout=0):
        super(ConvLSTM, self).__init__()
        
        # 配置参数
        self.input_channels = [input_channels] + hidden_channels  # 每层输入通道
        self.hidden_channels = hidden_channels                     # 每层隐藏通道
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)                    # 网络层数
        self.step = step                                          # 时间步数
        self.effective_step = effective_step                      # 有效输出步
        self._all_layers = []                                     # 所有层的列表
        self.pool_kernel_size = pool_kernel_size                   # 池化核大小
        self.conv_stride = conv_stride
        self.mp = nn.MaxPool2d(kernel_size=self.pool_kernel_size) # 最大池化层
        self.batch_norm = batch_normalization                      # 是否使用批归一化
        self.dropout_rate = dropout                                # dropout率
     
        # Dropout层
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        
        # 批归一化层列表（每层一个）
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_channels[i], eps=1e-05, momentum=0.1, affine=True)
            for i in range(self.num_layers)
        ])
        
        # 构建每层的ConvLSTMCell
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(
                self.input_channels[i], 
                self.hidden_channels[i], 
                self.kernel_size, 
                self.conv_stride
            )
            setattr(self, name, cell)  # 动态设置属性
            self._all_layers.append(cell)  # 添加到层列表
           

    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 时空序列输入, shape=(batch_size, seq_length, input_channels, H, W)
        
        Returns:
            tuple: (outputs, (x, new_c))
        """
        internal_state = []  # 存储各层的内部状态
        outputs = []         # 存储有效时间步的输出
        
        # 遍历每个时间步
        for step in range(self.step):
            # 获取当前时间步的输入
            x = input[:, step, :, :, :]
            
            # 遍历每一层
            for i in range(self.num_layers):
                # 获取当前层的名称
                name = 'cell{}'.format(i)
                
                # 第一层时间步需要初始化隐藏状态
                if step == 0:
                    bsize, channels, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize, 
                        hidden=self.hidden_channels[i],
                        shape=(height, width)
                    )
                    internal_state.append((h, c))

                # 获取上一层的状态
                (h, c) = internal_state[i]
                
                # 当前层前向传播
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)  # 更新状态
                
                # 应用Dropout（如果启用）
                if self.dropout_rate > 0:
                    x = self.dropout(x)
                
                # 应用批归一化（如果启用）
                if self.batch_norm:
                    x = self.bn_layers[i](x)
                
                # 应用最大池化
                x = self.mp(x)
                
            # 记录有效时间步的输出
            if step in self.effective_step:
                outputs.append(x)

        # 对最后时间步的细胞状态也应用池化
        new_c = self.mp(new_c)
        
        # 返回有效输出和最终状态
        return outputs, (x, new_c)


# 测试代码
if __name__ == '__main__':
    # 创建ConvLSTM实例
    convlstm = ConvLSTM(
        input_channels=3, 
        hidden_channels=[128, 64, 64, 32, 32], 
        kernel_size=3, 
        conv_stride=1,
        pool_kernel_size=(2, 2), 
        step=14, 
        effective_step=[13]
    )
    
    # 定义损失函数
    loss_fn = torch.nn.MSELoss()

    # 生成随机输入 (batch_size=2, seq_length=14, channels=3, height=244, width=244)
    input = Variable(torch.randn(2, 14, 3, 244, 244))
    target = Variable(torch.randn(2, 32, 244, 244)).double()

    # 前向传播
    output = convlstm(input)
    output = output[0][0].double()
    
    # 梯度检查
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print("Gradient check passed:", res)