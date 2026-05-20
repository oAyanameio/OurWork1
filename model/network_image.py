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
        self.intent_feature_dim = getattr(args, 'intent_feature_dim', 0)
        self.intent_proj = None
        if self.intent_feature_dim and self.intent_feature_dim > 0:
            self.intent_proj = nn.Sequential(
                nn.Linear(self.intent_feature_dim, args.hidden_size),
                nn.ReLU(),
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
            intent_dim = getattr(args, 'intent_feature_dim', 0)
            use_intent = intent_dim > 0
            cofe_kwargs = dict(
                input_size=self.size,
                hidden_size=args.cofe_hidden_size,
                num_layers=args.cofe_num_layers,
                use_intent=use_intent,
                intent_dim=intent_dim,
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
        self.fc_speed = nn.Linear(
            in_features=args.hidden_size, 
            out_features=self.size
        )
        
        self.hardtanh = nn.Hardtanh(
            min_val=-1 * args.hardtanh_limit, 
            max_val=args.hardtanh_limit
        )
        self.softmax = nn.Softmax(dim=1)
        self.cofe_loss_weight = getattr(args, 'cofe_loss_weight', 0.0)
        
        self.args = args

    def _make_hist_all_from_pos(self, pos, hist_yaw=None):
        if pos is None:
            return None
        if hist_yaw is None:
            hist_yaw = torch.zeros(*pos.shape[:-1], device=pos.device, dtype=pos.dtype)
        return torch.cat([pos, hist_yaw.unsqueeze(-1)], dim=-1)

    def _normalize_fpv_inputs(self, hist_all, hist_resnet=None, hist_seq_start_end=None):
        if hist_all is None:
            raise ValueError("T2FPV forward requires `hist_all` or batch-first `pos`.")
        input_len = getattr(self.args, 'input', None)
        if hist_all.dim() == 4:
            if input_len is not None and hist_all.shape[0] == input_len:
                T, B, N, F = hist_all.shape
                hist_all = hist_all.reshape(T, B * N, F)
                if hist_resnet is not None:
                    hist_resnet = hist_resnet.reshape(T, B * N, hist_resnet.shape[-1])
            else:
                B, T, N, F = hist_all.shape
                hist_all = hist_all.permute(1, 0, 2, 3).contiguous().reshape(T, B * N, F)
                if hist_resnet is not None:
                    hist_resnet = hist_resnet.permute(1, 0, 2, 3).contiguous().reshape(T, B * N, hist_resnet.shape[-1])
            if hist_seq_start_end is None:
                hist_seq_start_end = torch.tensor(
                    [[i * N, (i + 1) * N] for i in range(B)],
                    device=hist_all.device,
                    dtype=torch.long,
                )
        elif hist_all.dim() == 3:
            if input_len is not None and hist_all.shape[1] == input_len and hist_all.shape[0] != input_len:
                hist_all = hist_all.permute(1, 0, 2).contiguous()
                if hist_resnet is not None:
                    hist_resnet = hist_resnet.permute(1, 0, 2).contiguous()
            elif input_len is not None and hist_all.shape[0] == input_len:
                hist_all = hist_all.contiguous()
            else:
                hist_all = hist_all.contiguous()
            if hist_seq_start_end is None:
                hist_seq_start_end = torch.tensor([[0, hist_all.shape[1]]], device=hist_all.device, dtype=torch.long)
        else:
            raise ValueError(f"T2FPV hist_all must be 3D or 4D, got shape {tuple(hist_all.shape)}")

        if hist_all.shape[-1] < 3:
            yaw = torch.zeros(*hist_all.shape[:-1], device=hist_all.device, dtype=hist_all.dtype)
            hist_all = torch.cat([hist_all, yaw.unsqueeze(-1)], dim=-1)

        return hist_all, hist_resnet, hist_seq_start_end
        
    def forward(self, speed=None, pos=None, ped_attribute=None,
                ped_behavior=None, scene_attribute=None, images=None, optical=None, average=False, hist_all=None, hist_resnet=None, hist_seq_start_end=None, hist_yaw=None, hist_abs_gt=None, hist_yaw_gt=None, intent_feature=None, ego_idx=None):
        if self.args.dataset == 'T2FPV':
            if hist_all is None:
                hist_all = self._make_hist_all_from_pos(pos, hist_yaw)
            return self.forward_fpv(
                hist_all=hist_all,
                hist_resnet=hist_resnet,
                hist_seq_start_end=hist_seq_start_end,
                average=average,
                hist_abs_gt=hist_abs_gt,
                hist_yaw_gt=hist_yaw_gt,
                intent_feature=intent_feature,
                ego_idx=ego_idx,
            )

        cofe_loss = torch.zeros(1, device=self.args.device)
        if self.args.use_cofe and pos is not None:
            pos_seq_first = pos.permute(1, 0, 2)
            if self.training:
                if hist_abs_gt is not None:
                    gt_seq_first = hist_abs_gt.permute(1, 0, 2) if hist_abs_gt.dim() == 3 and hist_abs_gt.shape[1] == pos_seq_first.shape[0] else hist_abs_gt
                    gt_abs = gt_seq_first[..., :2]
                    gt_yaw = hist_yaw_gt if hist_yaw_gt is not None else None
                else:
                    gt_abs = pos_seq_first
                    gt_yaw = None
                cofe_loss = self.cofe.train_correction(
                    hist_abs_gt=gt_abs,
                    hist_yaw_gt=gt_yaw,
                    hist_abs_pred=pos_seq_first,
                    hist_yaw_pred=None,
                    hist_resnet=None,
                    hist_seq_start_end=None,
                    hist_intent=intent_feature,
                )
            corrected_pos_seq_first = self.cofe.infer_correction(
                pos_seq_first,
                hist_yaw_pred=None,
                hist_resnet=None,
                hist_seq_start_end=None,
                hist_intent=intent_feature,
            )
            pos = corrected_pos_seq_first.permute(1, 0, 2)
            new_speed = torch.zeros_like(pos)
            new_speed[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
            new_speed[:, 0, :] = new_speed[:, 1, :]
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
            optical = optical.view(batch_size_op * seq_len_op, c_op, h_op, w_op)
            op_feats = self.resnet(optical)
            op_feats = op_feats.view(batch_size_op, seq_len_op, -1)
            _, (himg_op, cimg_op) = self.op_encoder(op_feats)
            himg_op = himg_op[-1, :, :].squeeze(0)
            cimg_op = cimg_op[-1, :, :].squeeze(0)

        outputs = []
        if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
            outputs.append(ploss + sloss + pbloss + psloss + self.cofe_loss_weight * cofe_loss)
        else:
            outputs.append(ploss + sloss + pbloss + self.cofe_loss_weight * cofe_loss)
        outputs.append(cofe_loss)

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
        if self.intent_proj is not None and intent_feature is not None:
            intent_emb = self.intent_proj(intent_feature)
            hds = hds + intent_emb
            zds = zds + intent_emb

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()
            
        outputs.append(speed_outputs)

        return tuple(outputs)

    def forward_fpv(self, hist_all, hist_resnet=None, hist_seq_start_end=None, average=False, hist_abs_gt=None, hist_yaw_gt=None, intent_feature=None, ego_idx=None):
        hist_all, hist_resnet, hist_seq_start_end = self._normalize_fpv_inputs(
            hist_all, hist_resnet, hist_seq_start_end
        )
        T, N, _ = hist_all.shape
        device = hist_all.device

        if intent_feature is not None and self.intent_feature_dim > 0:
            assert intent_feature.shape[-1] == self.intent_feature_dim, \
                f"intent_feature last dim {intent_feature.shape[-1]} != intent_feature_dim {self.intent_feature_dim}"
            if intent_feature.dim() == 3:
                B, Na, D = intent_feature.shape
                assert B * Na == N, \
                    f"intent_feature [{B}, {Na}, {D}] flatten mismatch: B*Na={B*Na} != N={N}"
                intent_feature = intent_feature.reshape(N, D)
            elif intent_feature.dim() == 4:
                Ti, B, Na, D = intent_feature.shape
                assert B * Na == N, \
                    f"intent_feature [{Ti}, {B}, {Na}, {D}] flatten mismatch: B*Na={B*Na} != N={N}"
                intent_feature = intent_feature.reshape(Ti, N, D)

        hist_abs = hist_all[..., :2]
        hist_yaw = hist_all[..., 2]

        cofe_loss = torch.zeros(1, device=device)
        if self.args.use_cofe:
            if self.training:
                if hist_abs_gt is not None:
                    gt_all, _, _ = self._normalize_fpv_inputs(hist_abs_gt, None, hist_seq_start_end)
                    gt_abs = gt_all[..., :2]
                    if hist_yaw_gt is None:
                        gt_yaw = hist_yaw
                    elif hist_yaw_gt.dim() == 3:
                        gt_yaw = hist_yaw_gt.permute(1, 0, 2).contiguous().reshape(T, -1)
                    elif hist_yaw_gt.dim() == 2 and hist_yaw_gt.shape[0] != T and hist_yaw_gt.shape[1] == T:
                        gt_yaw = hist_yaw_gt.permute(1, 0).contiguous()
                    else:
                        gt_yaw = hist_yaw_gt
                else:
                    gt_abs = hist_abs
                    gt_yaw = hist_yaw
                cofe_loss = self.cofe.train_correction(
                    hist_abs_gt=gt_abs,
                    hist_yaw_gt=gt_yaw,
                    hist_abs_pred=hist_abs,
                    hist_yaw_pred=hist_yaw,
                    hist_resnet=hist_resnet,
                    hist_seq_start_end=hist_seq_start_end,
                    hist_intent=intent_feature,
                )
            corrected = self.cofe.infer_correction(
                hist_abs, hist_yaw, hist_resnet, hist_seq_start_end,
                hist_intent=intent_feature,
            )
        else:
            corrected = hist_abs
            cofe_loss = torch.zeros(1, device=device)

        pos = corrected.permute(1, 0, 2).contiguous()

        speed = torch.zeros_like(pos)
        speed[:, 1:] = pos[:, 1:] - pos[:, :-1]
        speed[:, 0] = speed[:, 1]

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

        outputs = [ploss + sloss + pbloss + psloss + self.cofe_loss_weight * cofe_loss, cofe_loss]

        speed_outputs = torch.tensor([], device=device)
        in_sp = speed[:, -1, :]

        hds = hpo + hsp + hpa + hsa + pb + him + hop
        zds = zpo + zsp + zpa + zsa + pb + cim + cop
        if self.intent_proj is not None and intent_feature is not None:
            if intent_feature.dim() == 3:
                intent_prior = intent_feature.mean(dim=0)
            else:
                intent_prior = intent_feature
            intent_emb = self.intent_proj(intent_prior)
            hds = hds + intent_emb
            zds = zds + intent_emb

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()

        if ego_idx is not None:
            ego_speed_preds = speed_outputs[ego_idx]
        else:
            ego_speed_preds = speed_outputs
        outputs.append(ego_speed_preds)

        return tuple(outputs)
