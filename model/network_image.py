"""
PTINet: Pedestrian Trajectory and Intention Prediction Network

This module implements the PTINet architecture, a multi-task learning framework
for joint pedestrian trajectory prediction and crossing intention prediction.
The model integrates multiple features including:
- Position and speed trajectories (bounding box sequences)
- Pedestrian attributes and behaviors
- Scene attributes
- Visual features (images via ConvLSTM or ResNet)
- Optical flow features

Reference: 
    PTINet paper - Joint Pedestrian Trajectory Prediction and Intention Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

# Import custom modules
from model.clstm import ConvLSTM
from model.vae import LSTMVAE
from model.cofe import CoFE


class PTINet(nn.Module):
    """
    PTINet主模型类
    
    架构概述:
    1. 多模态编码器模块：对位置、速度、行为、场景、图像、光流分别进行编码
    2. 特征融合模块：将各模态特征融合为统一的隐藏状态
    3. 双任务解码器：分别预测速度轨迹和穿越意图
    
    Args:
        args: 配置参数对象，包含以下关键参数：
            - dataset: 数据集名称 ('jaad', 'pie', 'titan')
            - hidden_size: 隐藏层维度
            - device: 计算设备 ('cuda' 或 'cpu')
            - use_attribute: 是否使用属性特征
            - use_image: 是否使用图像特征
            - image_network: 图像网络类型 ('clstm', 'resnet50', 'resnet18')
            - use_opticalflow: 是否使用光流特征
            - output: 输出时间步数
            - skip: 采样间隔
            - hardtanh_limit: Hardtanh激活函数的限制范围
    """
    
    def __init__(self, args):
        super(PTINet, self).__init__()

        # 根据数据集设置特征维度
        if args.dataset == 'jaad':
            self.size = 4                    # 位置/速度特征维度 (x, y, w, h)
            self.ped_attribute_size = 3      # 行人属性维度 (年龄、性别、群体大小)
            self.ped_behavior_size = 4       # 行为特征维度 (反应、手势、注视、点头)
            self.scene_attribute_size = 10   # 场景属性维度

        elif args.dataset == 'pie':
            self.size = 4
            self.ped_attribute_size = 2
            self.ped_behavior_size = 3
            self.scene_attribute_size = 4

        elif args.dataset == 'titan':
            self.size = 4
            self.ped_behavior_size = 3

        elif args.dataset == 'fpv':
            self.size = 2                    # 世界坐标 (x, y)
            self.ped_attribute_size = 3      # 默认值，FPV数据中不使用
            self.ped_behavior_size = 3       # 默认值，FPV数据中不使用
            self.scene_attribute_size = 4    # 默认值，FPV数据中不使用

        else:
            raise ValueError('Wrong dataset name!')

        # LSTM层数和隐变量维度
        self.num_layers = 1
        self.latent_size = args.hidden_size
        
        # ========== 时序编码器模块 ==========
        # 速度序列编码器 - 学习速度变化模式
        self.speed_encoder = LSTMVAE(
            input_size=self.size, 
            hidden_size=args.hidden_size, 
            latent_size=self.latent_size, 
            device=args.device
        )
        
        # 位置序列编码器 - 学习位置轨迹模式
        self.pos_encoder = LSTMVAE(
            input_size=self.size, 
            hidden_size=args.hidden_size, 
            latent_size=self.latent_size, 
            device=args.device
        )
        
        # ========== 属性编码器模块 ==========
        if args.use_attribute:
            # 行为特征编码器 - 学习行为模式
            self.ped_behavior_encoder = LSTMVAE(
                input_size=self.ped_behavior_size, 
                hidden_size=args.hidden_size, 
                latent_size=self.latent_size, 
                device=args.device
            )
            
            # 场景属性编码器 (JAAD和PIE数据集)
            if args.dataset == 'jaad' or args.dataset == 'pie':         
                self.scene_attribute_encoder = LSTMVAE(
                    input_size=self.scene_attribute_size, 
                    hidden_size=args.hidden_size, 
                    latent_size=self.latent_size, 
                    device=args.device
                )
                
                # MLP编码器 - 将行人属性(3维)映射到隐藏层维度
                self.mlp = nn.Sequential(
                    nn.Linear(self.ped_attribute_size, 64),   # 第一层：3->64
                    nn.ReLU(),                               # 激活函数
                    nn.Linear(64, args.hidden_size),         # 第二层：64->hidden_size
                    nn.ReLU()                                # 激活函数
                )

        # ========== 视觉编码器模块 ==========
        if args.use_image:
            if args.dataset == 'fpv':
                # FPV模式：使用预提取的ResNet特征 (T, N, 2048) 通过LSTM编码
                self.fpv_resnet_lstm = nn.LSTM(2048, args.hidden_size, batch_first=True)

            elif args.image_network == 'resnet50':
                # 使用ResNet50提取图像特征
                self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()  # 移除分类层
                self.img_encoder = LSTMVAE(
                    input_size=2048,  # ResNet50输出维度
                    hidden_size=args.hidden_size, 
                    latent_size=self.latent_size, 
                    device=args.device
                )

            elif args.image_network == 'resnet18':
                # 使用ResNet18提取图像特征
                self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder = nn.LSTM(
                    input_size=512,  # ResNet18输出维度
                    hidden_size=args.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True
                )
                
            elif args.image_network == 'clstm':
                # 使用ConvLSTM提取时空特征
                self.clstm = ConvLSTM(
                    input_channels=3,           # RGB图像
                    hidden_channels=[128, 64, 64, 32, 32],  # 5层卷积LSTM
                    kernel_size=3,              # 卷积核大小
                    conv_stride=1,              # 卷积步长
                    pool_kernel_size=(2, 2),    # 池化核大小
                    step=5,                     # 时间步数
                    effective_step=[4]          # 有效输出步
                )
                # 自适应池化和全连接层将特征映射到hidden_size
                self.pooling_h = nn.AdaptiveAvgPool2d((1, 1))
                self.pooling_c = nn.AdaptiveAvgPool2d((1, 1))
                self.linear_c = nn.Linear(in_features=32, out_features=512)
                self.linear_h = nn.Linear(in_features=32, out_features=512)

        # ========== 光流编码器模块 ==========
        if args.use_opticalflow:
            # 修改ResNet以支持4通道输入（光流x,y方向各2通道）
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Identity()
            self.op_encoder = nn.LSTM(
                input_size=2048, 
                hidden_size=args.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )

        # ========== CoFE轨迹去噪修复模块 ==========
        if args.use_cofe:
            cofe_kwargs = dict(
                input_size=self.size,
                hidden_size=args.cofe_hidden_size,
                num_layers=args.cofe_num_layers,
            )
            if args.dataset == 'fpv':
                cofe_kwargs.update(dict(
                    use_resnet=getattr(args, 'cofe_use_resnet', False),
                    no_abs=True,
                    idxs=[6, 7],
                ))
            self.cofe = CoFE(**cofe_kwargs)

        # ========== 解码器模块 ==========
        # 位置嵌入层 - 将隐藏状态映射回位置空间
        self.pos_embedding = nn.Sequential(
            nn.Linear(in_features=args.hidden_size, out_features=self.size),
            nn.ReLU()
        )
        
        # LSTMCell解码器 - 用于序列生成
        self.speed_decoder = nn.LSTMCell(      # 速度轨迹解码器
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        self.crossing_decoder = nn.LSTMCell(   # 穿越意图解码器
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        self.attrib_decoder = nn.LSTMCell(     # 属性解码器（备用）
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        
        # 全连接输出层
        self.fc_speed = nn.Linear(             # 速度预测头
            in_features=args.hidden_size, 
            out_features=self.size
        )
        self.fc_crossing = nn.Sequential(      # 穿越意图分类头
            nn.Linear(in_features=args.hidden_size, out_features=2),
            nn.ReLU()
        )
        self.fc_attrib = nn.Sequential(        # 属性预测头（备用）
            nn.Linear(in_features=args.hidden_size, out_features=3),
            nn.ReLU()
        )
        
        # 激活函数
        self.hardtanh = nn.Hardtanh(           # 限制速度输出范围
            min_val=-1 * args.hardtanh_limit, 
            max_val=args.hardtanh_limit
        )
        self.softmax = nn.Softmax(dim=1)       # 意图概率归一化
        
        # 保存配置参数
        self.args = args
        
    def forward(self, speed=None, pos=None, ped_attribute=None, 
                ped_behavior=None, scene_attribute=None, images=None, 
                optical=None, average=False, hist_all=None, 
                hist_resnet=None, hist_seq_start_end=None):
        """
        前向传播函数
        
        Args:
            speed: 速度序列, shape=(batch_size, input_len-1, 4)
            pos: 位置序列, shape=(batch_size, input_len, 4)
            ped_attribute: 行人属性, shape=(batch_size, 3)
            ped_behavior: 行为序列, shape=(batch_size, input_len, 4)
            scene_attribute: 场景属性序列, shape=(batch_size, input_len, 10)
            images: 图像序列, shape=(batch_size, input_len, 3, H, W)
            optical: 光流序列, shape=(batch_size, input_len, 4, H, W)
            average: 是否计算平均意图标签
            hist_all: (FPV模式) 多agent历史数据, shape=(timesteps, num_agents, 7)
            hist_resnet: (FPV模式) 预提取ResNet特征, shape=(timesteps, num_agents, 2048)
            hist_seq_start_end: (FPV模式) 场景边界索引, shape=(num_scenes, 2)
            
        Returns:
            tuple: 包含以下元素的元组：
                [0] - 总重构损失
                [1] - 速度预测输出, shape=(batch_size, output_len, size)
                [2] - 穿越意图预测, shape=(batch_size, output_len, 2)
                [3] - (可选)平均意图标签, shape=(batch_size,)
        """

        # FPV数据流调度
        if self.args.dataset == 'fpv':
            return self.forward_fpv(
                hist_all=hist_all,
                hist_resnet=hist_resnet,
                hist_seq_start_end=hist_seq_start_end,
                average=average
            )

        # ========== 0. CoFE轨迹修复（可选） ==========
        # 流程: batch_first -> seq_first -> CoFE去噪 -> batch_first -> LSTM-VAE
        if self.args.use_cofe and pos is not None:
            # 步骤1: 维度转换 [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
            #         CoFE内部使用for t in range(seq_len)按时间步循环，要求时间步在第0维
            pos_seq_first = pos.permute(1, 0, 2)
            # 步骤2: 送入CoFE进行去噪修复
            #         PTINet为BEV视角，没有yaw/ResNet/seq_start_end数据，传None
            corrected_pos_seq_first = self.cofe.infer_correction(
                pos_seq_first,
                hist_yaw_pred=None,
                hist_resnet=None,
                hist_seq_start_end=None,
            )
            # 步骤3: 维度复原 [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
            #         恢复batch_first格式，供后续LSTMVAE(batch_first=True)使用
            pos = corrected_pos_seq_first.permute(1, 0, 2)
            # 步骤4: 从修复后的位置序列重新计算速度（帧间差）
            #         保证位置与速度的物理一致性
            # 4a: 废弃原speed，创建与pos形状相同的新张量
            new_speed = torch.zeros_like(pos)
            # 4b: 一阶差分计算：第t步速度 = pos[t] - pos[t-1]
            #     第0步保持为0（无前一帧可差分）
            new_speed[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
            # 4c: 覆盖原speed变量
            speed = new_speed

        # ========== 1. 时序特征编码 ==========
        # 速度编码
        sloss, _, zsp, hsp, _ = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)      # 隐藏状态: (batch_size, hidden_size)
        zsp = torch.mean(zsp, axis=1) # 隐变量时间平均: (batch_size, latent_size)
        
        # 位置编码
        ploss, _, zpo, hpo, _ = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)      # 隐藏状态: (batch_size, hidden_size)
        zpo = torch.mean(zpo, axis=1) # 隐变量时间平均: (batch_size, latent_size)

        # ========== 2. 属性特征编码 ==========
        if self.args.use_attribute:
            # 行为特征编码
            pbloss, _, zpa, hpa, _ = self.ped_behavior_encoder(ped_behavior)
            hpa = hpa[0].squeeze(0)
            zpa = torch.mean(zpa, axis=1)

            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                # 场景属性编码
                psloss, _, zsa, hsa, _ = self.scene_attribute_encoder(scene_attribute)
                hsa = hsa[0].squeeze(0)
                zsa = torch.mean(zsa, axis=1)

                # 行人属性MLP编码
                pb = self.mlp(ped_attribute)

        # ========== 3. 视觉特征编码 ==========
        if self.args.use_image:
            batch_size, seq_len, c, h, w = images.size()

            if self.args.image_network == 'clstm':
                # ConvLSTM处理时序图像
                _, (himg, cimg) = self.clstm(images)
                # 自适应池化 + 全连接映射
                himg = self.pooling_h(himg).view(himg.size(0), -1)
                himg = self.linear_h(himg)
                cimg = self.pooling_c(cimg).view(cimg.size(0), -1)
                cimg = self.linear_c(cimg)
            else:
                # ResNet处理
                images = images.view(batch_size * seq_len, c, h, w)
                img_feats = self.resnet(images)
                img_feats = img_feats.view(batch_size, seq_len, -1)
                imgloss, _, zim, him, _ = self.img_encoder(img_feats)
                him = him[0].squeeze(0)
                zim = torch.mean(zim, axis=1)

        # ========== 4. 光流特征编码 ==========
        if self.args.use_opticalflow:
            batch_size_op, seq_len_op, c_op, h_op, w_op = optical.size()
            optical = optical.view(batch_size * seq_len_op, c_op, h_op, w_op)
            op_feats = self.resnet(optical)
            op_feats = op_feats.view(batch_size, seq_len_op, -1)
            _, (himg_op, cimg_op) = self.op_encoder(op_feats)
            himg_op = himg_op[-1, :, :].squeeze(0)
            cimg_op = cimg_op[-1, :, :].squeeze(0)

        # ========== 5. 计算总重构损失 ==========
        outputs = []
        if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
            outputs.append(ploss + sloss + pbloss + psloss)
        else:
            outputs.append(ploss + sloss + pbloss)

        # ========== 6. 速度轨迹预测 ==========
        speed_outputs = torch.tensor([], device=self.args.device)
        in_sp = speed[:, -1, :]  # 初始输入：最后一帧的速度
        
        # 特征融合 - 速度预测分支
        hds = hpo + hsp  # 位置 + 速度特征
        zds = zpo + zsp

        if self.args.use_attribute:
            hds = hds + hpa  
            zds = zds + zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                hds = hds + hsa + hpa + pb 
                zds = zds + zpa + zsa + pb

        if self.args.use_image:
            hds = hds + himg
            zds = zds + cimg 

        if self.args.use_opticalflow:
            hds = hds + himg_op
            zds = zds + cimg_op 

        # 多步预测
        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))  # 限制输出范围
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()  # 截断梯度，避免梯度爆炸
            
        outputs.append(speed_outputs)

        # ========== 7. 穿越意图预测 ==========
        crossing_outputs = torch.tensor([], device=self.args.device)
        in_cr = pos[:, -1, :]  # 初始输入：最后一帧的位置
        
        # 特征融合 - 意图预测分支（重新计算以保持独立性）
        hdc = hpo + hsp
        zdc = zpo + zsp

        if self.args.use_attribute:
            hdc = hdc + hpa  
            zdc = zdc + zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
                hdc = hdc + hsa + hpa + pb 
                zdc = zdc + zpa + zsa + pb

        if self.args.use_image:
            hdc = hdc + himg
            zdc = zdc + cimg 

        if self.args.use_opticalflow:
            hdc = hdc + himg_op
            zdc = zdc + cimg_op 

        # 多步意图预测
        for i in range(self.args.output // self.args.skip):
            hdc, zdc = self.crossing_decoder(in_cr, (hdc, zdc))
            crossing_output = self.fc_crossing(hdc)
            in_cr = self.pos_embedding(hdc).detach()  # 更新输入
            crossing_output = self.softmax(crossing_output)  # 归一化为概率
            crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim=1)

        outputs.append(crossing_outputs)
        
        # 计算平均意图标签（用于评估）
        if average:
            crossing_labels = torch.argmax(crossing_outputs, dim=2)  # 每帧预测标签
            intention = torch.max(crossing_labels, dim=1)[0]        # 取最大投票
            outputs.append(intention)
        
        return tuple(outputs)

    def forward_fpv(self, hist_all, hist_resnet=None, hist_seq_start_end=None, average=False):
        """
        FPV模式前向传播

        输入FPV数据集格式的多agent轨迹数据，执行完整预测流程：
        1. CoFE轨迹去噪修复（使用yaw编码和ego相对距离）
        2. LSTMVAE编码（位置+速度）
        3. 缺失模态填充（属性/行为/场景/光流 → 零张量）
        4. ResNet视觉特征编码（通过fpv_resnet_lstm）
        5. 特征融合与双任务解码

        Args:
            hist_all: 多agent历史数据, shape=(timesteps, num_agents, 7)
                      [x_world, y_world, yaw, img_x, img_y, valid, agent_id]
            hist_resnet: 预提取ResNet特征, shape=(timesteps, num_agents, 2048)
            hist_seq_start_end: 场景边界索引, shape=(num_scenes, 2)
            average: 是否计算平均意图标签

        Returns:
            tuple: (loss, speed_outputs, crossing_outputs, [intention])
        """
        T, N, _ = hist_all.shape
        device = hist_all.device

        # 1. 拆分FPV数据
        hist_abs = hist_all[..., :2]   # (T, N, 2) 世界坐标
        hist_yaw = hist_all[..., 2]    # (T, N) 偏航角

        # 2. CoFE轨迹修正（完整FPV版：含yaw编码、ego_dists、ResNet融合）
        if self.args.use_cofe:
            corrected = self.cofe.infer_correction(
                hist_abs, hist_yaw, hist_resnet, hist_seq_start_end
            )
        else:
            corrected = hist_abs

        # 3. 维度转换 (T, N, 2) → (N, T, 2) batch_first格式
        pos = corrected.permute(1, 0, 2).contiguous()

        # 4. 速度重计算（物理一致性）
        speed = torch.zeros_like(pos)
        speed[:, 1:] = pos[:, 1:] - pos[:, :-1]

        # 5. LSTMVAE编码
        sloss, _, zsp, hsp, _ = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp = torch.mean(zsp, axis=1)

        ploss, _, zpo, hpo, _ = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo = torch.mean(zpo, axis=1)

        # 6. 缺失模态 → 零张量（FPV数据不含这些模态）
        batch = N
        hidden_size = self.args.hidden_size

        hpa = torch.zeros(batch, hidden_size, device=device)
        hsa = torch.zeros(batch, hidden_size, device=device)
        zpa = torch.zeros(batch, hidden_size, device=device)
        zsa = torch.zeros(batch, hidden_size, device=device)
        pb = torch.zeros(batch, hidden_size, device=device)
        pbloss = torch.zeros(1, device=device)
        psloss = torch.zeros(1, device=device)

        # 7. ResNet视觉特征编码
        if self.args.use_image and hist_resnet is not None and hasattr(self, 'fpv_resnet_lstm'):
            resnet_feat = hist_resnet.permute(1, 0, 2)
            _, (h_fpv_im, c_fpv_im) = self.fpv_resnet_lstm(resnet_feat)
            him = h_fpv_im[-1].squeeze(0)
            cim = c_fpv_im[-1].squeeze(0)
        else:
            him = torch.zeros(batch, hidden_size, device=device)
            cim = torch.zeros(batch, hidden_size, device=device)

        # 光流（FPV不含）
        hop = torch.zeros(batch, hidden_size, device=device)
        cop = torch.zeros(batch, hidden_size, device=device)

        # 8. 总重构损失
        outputs = [ploss + sloss + pbloss + psloss]

        # 9. 速度轨迹预测
        speed_outputs = torch.tensor([], device=device)
        in_sp = speed[:, -1, :]

        hds = hpo + hsp + hpa + hsa + pb + him + hop
        zds = zpo + zpa + zsa + pb + cim + cop

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()

        outputs.append(speed_outputs)

        # 10. 穿越意图预测
        crossing_outputs = torch.tensor([], device=device)
        in_cr = pos[:, -1, :]

        hdc = hpo + hsp + hpa + hsa + pb + him + hop
        zdc = zpo + zpa + zsa + pb + cim + cop

        for i in range(self.args.output // self.args.skip):
            hdc, zdc = self.crossing_decoder(in_cr, (hdc, zdc))
            crossing_output = self.fc_crossing(hdc)
            in_cr = self.pos_embedding(hdc).detach()
            crossing_output = self.softmax(crossing_output)
            crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim=1)

        outputs.append(crossing_outputs)

        if average:
            crossing_labels = torch.argmax(crossing_outputs, dim=2)
            intention = torch.max(crossing_labels, dim=1)[0]
            outputs.append(intention)

        return tuple(outputs)