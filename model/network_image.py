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

        if args.dataset == 'jaad':
            self.size = 4
            self.ped_attribute_size = 3
            self.ped_behavior_size = 4
            self.scene_attribute_size = 10

        elif args.dataset == 'pie':
            self.size = 4
            self.ped_attribute_size = 2
            self.ped_behavior_size = 3
            self.scene_attribute_size = 4

        elif args.dataset == 'titan':
            self.size = 4
            self.ped_behavior_size = 3

        elif args.dataset == 'T2FPV':
            self.size = 2
            self.ped_attribute_size = 3
            self.ped_behavior_size = 3
            self.scene_attribute_size = 4

        else:
            raise ValueError('Wrong dataset name!')

        self.num_layers = 1
        self.latent_size = args.hidden_size
        
        self.speed_encoder = LSTMVAE(
            input_size=self.size, 
            hidden_size=args.hidden_size, 
            latent_size=self.latent_size, 
            device=args.device
        )
        
        self.pos_encoder = LSTMVAE(
            input_size=self.size, 
            hidden_size=args.hidden_size, 
            latent_size=self.latent_size, 
            device=args.device
        )
        
        if args.use_attribute:
            self.ped_behavior_encoder = LSTMVAE(
                input_size=self.ped_behavior_size, 
                hidden_size=args.hidden_size, 
                latent_size=self.latent_size, 
                device=args.device
            )
            
            if args.dataset == 'jaad' or args.dataset == 'pie':         
                self.scene_attribute_encoder = LSTMVAE(
                    input_size=self.scene_attribute_size, 
                    hidden_size=args.hidden_size, 
                    latent_size=self.latent_size, 
                    device=args.device
                )
                
                self.mlp = nn.Sequential(
                    nn.Linear(self.ped_attribute_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, args.hidden_size),
                    nn.ReLU()
                )

        if args.use_image:
            if args.dataset == 'T2FPV':
                self.fpv_resnet_lstm = nn.LSTM(2048, args.hidden_size, batch_first=True)

            elif args.image_network == 'resnet50':
                self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder = LSTMVAE(
                    input_size=2048,
                    hidden_size=args.hidden_size, 
                    latent_size=self.latent_size, 
                    device=args.device
                )

            elif args.image_network == 'resnet18':
                self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder = nn.LSTM(
                    input_size=512,
                    hidden_size=args.hidden_size,
                    num_layers=self.num_layers,batch_first=True
                )
                
            elif args.image_network == 'clstm':
                self.clstm = ConvLSTM(
                    input_channels=3,
                    hidden_channels=[128, 64, 64, 32, 32],
                    kernel_size=3,
                    conv_stride=1,
                    pool_kernel_size=(2, 2),
                    step=5,
                    effective_step=[4]
                )
                self.pooling_h = nn.AdaptiveAvgPool2d((1, 1))
                self.pooling_c = nn.AdaptiveAvgPool2d((1, 1))
                self.linear_c = nn.Linear(in_features=32, out_features=512)
                self.linear_h = nn.Linear(in_features=32, out_features=512)

        if args.use_opticalflow:
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Identity()
            self.op_encoder = nn.LSTM(
                input_size=2048, 
                hidden_size=args.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )

        if args.use_cofe:
            cofe_kwargs = dict(
                input_size=self.size,
                hidden_size=args.cofe_hidden_size,
                num_layers=args.cofe_num_layers,
            )
            if args.dataset == 'T2FPV':
                cofe_kwargs.update(dict(
                    use_resnet=getattr(args, 'cofe_use_resnet', False),
                    no_abs=True,
                    idxs=[6, 7],
                ))
            self.cofe = CoFE(**cofe_kwargs)

        self.pos_embedding = nn.Sequential(
            nn.Linear(in_features=args.hidden_size, out_features=self.size),
            nn.ReLU()
        )
        
        self.speed_decoder = nn.LSTMCell(
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        self.crossing_decoder = nn.LSTMCell(
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        self.attrib_decoder = nn.LSTMCell(
            input_size=self.size, 
            hidden_size=args.hidden_size
        )
        
        self.fc_speed = nn.Linear(
            in_features=args.hidden_size, 
            out_features=self.size
        )
        self.fc_crossing = nn.Sequential(
            nn.Linear(in_features=args.hidden_size, out_features=2),
            nn.ReLU()
        )
        self.fc_attrib = nn.Sequential(
            nn.Linear(in_features=args.hidden_size, out_features=3),
            nn.ReLU()
        )
        
        self.hardtanh = nn.Hardtanh(
            min_val=-1 * args.hardtanh_limit, 
            max_val=args.hardtanh_limit
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.args = args
        
    def forward(self, speed=None, pos=None, ped_attribute=None, 
                ped_behavior=None, scene_attribute=None, images=None, optical=None, average=False, hist_all=None, hist_resnet=None, hist_seq_start_end=None):
        if self.args.dataset == 'T2FPV':
            return self.forward_fpv(
                hist_all=hist_all,
                hist_resnet=hist_resnet,
                hist_seq_start_end=hist_seq_start_end,
                average=average
            )

        if self.args.use_cofe and pos is not None:
            pos_seq_first = pos.permute(1, 0, 2)
            corrected_pos_seq_first = self.cofe.infer_correction(
                pos_seq_first,
                hist_yaw_pred=None,
                hist_resnet=None,
                hist_seq_start_end=None,
            )
            pos = corrected_pos_seq_first.permute(1, 0, 2)
            new_speed = torch.zeros_like(pos)
            new_speed[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
            speed = new_speed

        pbloss = torch.zeros(1, device=self.args.device)
        psloss = torch.zeros(1, device=self.args.device)
        hpa = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        zpa = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        hsa = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        zsa = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        pb = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        himg = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        cimg = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        himg_op = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)
        cimg_op = torch.zeros(pos.size(0), self.args.hidden_size, device=self.args.device)

        sloss, _, zsp, hsp, _ = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp = torch.mean(zsp, axis=1)
        
        ploss, _, zpo, hpo, _ = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo = torch.mean(zpo, axis=1)

        if self.args.use_attribute:
            pbloss, _, zpa, hpa, _ = self.ped_behavior_encoder(ped_behavior)
            hpa = hpa[0].squeeze(0)
            zpa = torch.mean(zpa, axis=1)

            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                psloss, _, zsa, hsa, _ = self.scene_attribute_encoder(scene_attribute)
                hsa = hsa[0].squeeze(0)
                zsa = torch.mean(zsa, axis=1)

                pb = self.mlp(ped_attribute)

        if self.args.use_image:
            batch_size, seq_len, c, h, w = images.size()

            if self.args.image_network == 'clstm':
                _, (himg, cimg) = self.clstm(images)
                himg = self.pooling_h(himg).view(himg.size(0), -1)
                himg = self.linear_h(himg)
                cimg = self.pooling_c(cimg).view(cimg.size(0), -1)
                cimg = self.linear_c(cimg)
            else:
                images = images.view(batch_size * seq_len, c, h, w)
                img_feats = self.resnet(images)
                img_feats = img_feats.view(batch_size, seq_len, -1)
                imgloss, _, zim, him, _ = self.img_encoder(img_feats)
                himg = him[0].squeeze(0)
                cimg = torch.mean(zim, axis=1)

        if self.args.use_opticalflow:
            batch_size_op, seq_len_op, c_op, h_op, w_op = optical.size()
            optical = optical.view(batch_size * seq_len_op, c_op, h_op, w_op)
            op_feats = self.resnet(optical)
            op_feats = op_feats.view(batch_size, seq_len_op, -1)
            _, (himg_op, cimg_op) = self.op_encoder(op_feats)
            himg_op = himg_op[-1, :, :].squeeze(0)
            cimg_op = cimg_op[-1, :, :].squeeze(0)

        outputs = []
        if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
            outputs.append(ploss + sloss + pbloss + psloss)
        else:
            outputs.append(ploss + sloss + pbloss)

        speed_outputs = torch.tensor([], device=self.args.device)
        in_sp = speed[:, -1, :]
        
        hds = hpo + hsp
        zds = zpo + zsp

        if self.args.use_attribute:
            hds = hds + hpa
            zds = zds + zpa
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                hds = hds + hsa + pb
                zds = zds + zsa + pb

        if self.args.use_image:
            hds = hds + himg
            zds = zds + cimg

        if self.args.use_opticalflow:
            hds = hds + himg_op
            zds = zds + cimg_op

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()
            
        outputs.append(speed_outputs)

        crossing_outputs = torch.tensor([], device=self.args.device)
        in_cr = pos[:, -1, :]
        
        hdc = hpo + hsp
        zdc = zpo + zsp

        if self.args.use_attribute:
            hdc = hdc + hpa
            zdc = zdc + zpa
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
                hdc = hdc + hsa + pb
                zdc = zdc + zsa + pb

        if self.args.use_image:
            hdc = hdc + himg
            zdc = zdc + cimg

        if self.args.use_opticalflow:
            hdc = hdc + himg_op
            zdc = zdc + cimg_op

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

    def forward_fpv(self, hist_all, hist_resnet=None, hist_seq_start_end=None, average=False):
        T, N, _ = hist_all.shape
        device = hist_all.device

        hist_abs = hist_all[..., :2]
        hist_yaw = hist_all[..., 2]

        if self.args.use_cofe:
            corrected = self.cofe.infer_correction(
                hist_abs, hist_yaw, hist_resnet, hist_seq_start_end
            )
        else:
            corrected = hist_abs

        pos = corrected.permute(1, 0, 2).contiguous()

        speed = torch.zeros_like(pos)
        speed[:, 1:] = pos[:, 1:] - pos[:, :-1]

        sloss, _, zsp, hsp, _ = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp = torch.mean(zsp, axis=1)

        ploss, _, zpo, hpo, _ = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo = torch.mean(zpo, axis=1)

        batch = N
        hidden_size = self.args.hidden_size

        hpa = torch.zeros(batch, hidden_size, device=device)
        hsa = torch.zeros(batch, hidden_size, device=device)
        zpa = torch.zeros(batch, hidden_size, device=device)
        zsa = torch.zeros(batch, hidden_size, device=device)
        pb = torch.zeros(batch, hidden_size, device=device)
        pbloss = torch.zeros(1, device=device)
        psloss = torch.zeros(1, device=device)

        if self.args.use_image and hist_resnet is not None and hasattr(self, 'fpv_resnet_lstm'):
            resnet_feat = hist_resnet.permute(1, 0, 2)
            _, (h_fpv_im, c_fpv_im) = self.fpv_resnet_lstm(resnet_feat)
            him = h_fpv_im[-1].squeeze(0)
            cim = c_fpv_im[-1].squeeze(0)
        else:
            him = torch.zeros(batch, hidden_size, device=device)
            cim = torch.zeros(batch, hidden_size, device=device)

        hop = torch.zeros(batch, hidden_size, device=device)
        cop = torch.zeros(batch, hidden_size, device=device)

        outputs = [ploss + sloss + pbloss + psloss]

        speed_outputs = torch.tensor([], device=device)
        in_sp = speed[:, -1, :]

        hds = hpo + hsp + hpa + hsa + pb + him + hop
        zds = zpo + zsp + zpa + zsa + pb + cim + cop

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()

        outputs.append(speed_outputs)

        crossing_outputs = torch.tensor([], device=device)
        in_cr = pos[:, -1, :]

        hdc = hpo + hsp + hpa + hsa + pb + him + hop
        zdc = zpo + zsp + zpa + zsa + pb + cim + cop

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
