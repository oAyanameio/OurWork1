import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from models.clstm import ConvLSTM
from models.vae import LSTMVAE
from models.cofe import CoFE


class PTINet(nn.Module):
    def __init__(self, args):
        super(PTINet, self).__init__()

        self.size = 2
        self.ped_attribute_size = 3
        self.ped_behavior_size = 3
        self.scene_attribute_size = 4
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

        lcf_dim = getattr(args, 'lcf_feature_dim', 0)
        self.lcf_proj = None
        if lcf_dim > 0:
            self.lcf_proj = nn.Sequential(
                nn.Linear(lcf_dim, args.hidden_size),
                nn.ReLU(),
            )

        self.intent_feature_dim = getattr(args, 'intent_feature_dim', 0)
        self.intent_proj = None
        if self.intent_feature_dim > 0:
            self.intent_proj = nn.Sequential(
                nn.Linear(self.intent_feature_dim, args.hidden_size),
                nn.ReLU(),
            )

        if args.use_image:
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
            self.optical_resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.optical_resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.optical_resnet.fc = nn.Identity()
            self.op_encoder = nn.LSTM(
                input_size=2048,
                hidden_size=args.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )

        if args.use_cofe:
            intent_dim = getattr(args, 'intent_feature_dim', 0)
            self.cofe = CoFE(
                input_size=self.size,
                hidden_size=args.cofe_hidden_size,
                num_layers=args.cofe_num_layers,
                use_intent=intent_dim > 0,
                intent_dim=intent_dim,
                use_resnet=getattr(args, 'cofe_use_resnet', False),
                no_abs=True,
                idxs=[6, 7],
                dropout=getattr(args, 'cofe_dropout', 0.3),
                layer_norm=getattr(args, 'cofe_layer_norm', False),
            )

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
        self.cofe_frozen = getattr(args, 'cofe_frozen', False)

        self.args = args
        self._seq_start_end_cache = {}

    def _get_cached_seq_start_end(self, B, N, device):
        key = (B, N)
        cached = self._seq_start_end_cache.get(key)
        if cached is None:
            offsets = torch.arange(0, B * N, N, device='cpu', dtype=torch.long)
            cached = torch.stack([offsets, offsets + N], dim=1)
            self._seq_start_end_cache[key] = cached
        return cached.to(device)

    def _make_hist_all_from_pos(self, pos, hist_yaw=None):
        if pos is None:
            return None
        if hist_yaw is None:
            hist_yaw = torch.zeros(*pos.shape[:-1], device=pos.device, dtype=pos.dtype)
        return torch.cat([pos, hist_yaw.unsqueeze(-1)], dim=-1)

    def _normalize_fpv_inputs_legacy_3d(self, hist_all, hist_resnet, hist_seq_start_end):
        import warnings
        warnings.warn(
            "_normalize_fpv_inputs received 3D input, normalize to [B, T, N, F] 4D format."
            " Using legacy compatibility path.",
            DeprecationWarning, stacklevel=3
        )
        input_len = getattr(self.args, 'input', None)
        if input_len is not None and hist_all.shape[1] == input_len and hist_all.shape[0] != input_len:
            hist_all = hist_all.permute(1, 0, 2).contiguous()
            if hist_resnet is not None:
                hist_resnet = hist_resnet.permute(1, 0, 2).contiguous()
        elif input_len is not None and hist_all.shape[0] == input_len:
            hist_all = hist_all.contiguous()
        else:
            hist_all = hist_all.contiguous()
        if hist_seq_start_end is None:
            hist_seq_start_end = torch.tensor(
                [[0, hist_all.shape[1]]], device=hist_all.device, dtype=torch.long
            )
        return hist_all, hist_resnet, hist_seq_start_end

    def _normalize_fpv_inputs(self,
                              hist_all,
                              hist_resnet=None, hist_seq_start_end=None):
        if hist_all is None:
            raise ValueError("T2FPV forward requires `hist_all` or batch-first `pos`.")
        if hist_all.dim() == 4:
            B, T, N, F_dim = hist_all.shape
            hist_all = hist_all.permute(1, 0, 2, 3).reshape(T, B * N, F_dim)
            if hist_resnet is not None:
                hist_resnet = hist_resnet.permute(1, 0, 2, 3).reshape(T, B * N, hist_resnet.shape[-1])
            if hist_seq_start_end is None:
                hist_seq_start_end = self._get_cached_seq_start_end(B, N, hist_all.device)
        elif hist_all.dim() == 3:
            hist_all, hist_resnet, hist_seq_start_end = self._normalize_fpv_inputs_legacy_3d(
                hist_all, hist_resnet, hist_seq_start_end
            )
        else:
            raise ValueError(
                f"T2FPV hist_all must be 3D or 4D, got shape {tuple(hist_all.shape)}"
            )

        if hist_all.shape[-1] < 3:
            hist_all = F.pad(hist_all, (0, 3 - hist_all.shape[-1]))

        return hist_all, hist_resnet, hist_seq_start_end

    def _resolve_fpv_gt_yaw(self, hist_yaw_gt, default_yaw, T):
        if hist_yaw_gt is None:
            return default_yaw
        elif hist_yaw_gt.dim() == 3:
            return hist_yaw_gt.permute(1, 0, 2).contiguous().reshape(T, -1)
        elif hist_yaw_gt.dim() == 2 and hist_yaw_gt.shape[0] != T and hist_yaw_gt.shape[1] == T:
            return hist_yaw_gt.permute(1, 0).contiguous()
        else:
            return hist_yaw_gt

    def forward(self,
                speed=None,
                pos=None,
                ped_attribute=None,
                ped_behavior=None,
                scene_attribute=None,
                images=None,
                optical=None,
                lcf_features=None,
                average=False,
                hist_all=None,
                hist_resnet=None,
                hist_seq_start_end=None,
                hist_yaw=None,
                hist_abs_gt=None,
                hist_yaw_gt=None,
                intent_feature=None,
                ego_idx=None):
        if hist_all is not None:
            hist_all, hist_resnet, hist_seq_start_end = self._normalize_fpv_inputs(
                hist_all, hist_resnet, hist_seq_start_end
            )
            T, N, _ = hist_all.shape
            device = hist_all.device

            if intent_feature is not None and self.intent_feature_dim > 0:
                assert intent_feature.shape[-1] == self.intent_feature_dim
                if intent_feature.dim() == 3:
                    B_i, Na, D = intent_feature.shape
                    intent_feature = intent_feature.reshape(N, D)
                elif intent_feature.dim() == 4:
                    Ti, B_i, Na, D = intent_feature.shape
                    intent_feature = intent_feature.reshape(Ti, N, D)

            hist_abs = hist_all[..., :2]
            hist_yaw_fpv = hist_all[..., 2]

            cofe_loss = torch.zeros(1, device=device)
            if self.args.use_cofe:
                if self.training and not self.cofe_frozen:
                    if hist_abs_gt is not None:
                        gt_all, _, _ = self._normalize_fpv_inputs(
                            hist_abs_gt, None, hist_seq_start_end
                        )
                        gt_abs = gt_all[..., :2]
                        gt_yaw = self._resolve_fpv_gt_yaw(hist_yaw_gt, hist_yaw_fpv, T)
                    else:
                        gt_abs, gt_yaw = hist_abs, hist_yaw_fpv
                    cofe_loss = self.cofe.train_correction(
                        hist_abs_gt=gt_abs,
                        hist_yaw_gt=gt_yaw,
                        hist_abs_pred=hist_abs,
                        hist_yaw_pred=hist_yaw_fpv,
                        hist_resnet=hist_resnet,
                        hist_seq_start_end=hist_seq_start_end,
                        hist_intent=intent_feature,
                    )
                corrected = self.cofe.infer_correction(
                    hist_abs, hist_yaw_fpv, hist_resnet, hist_seq_start_end,
                    hist_intent=intent_feature,
                )
            else:
                corrected = hist_abs
                cofe_loss = torch.zeros(1, device=device)

            pos = corrected.permute(1, 0, 2).contiguous()
        else:
            device = self.args.device

            cofe_loss = torch.zeros(1, device=device)
            if self.args.use_cofe and pos is not None:
                pos_seq_first = pos.permute(1, 0, 2)
                if self.training and not self.cofe_frozen:
                    if hist_abs_gt is not None:
                        gt_seq = (
                            hist_abs_gt.permute(1, 0, 2)
                            if hist_abs_gt.dim() == 3 and hist_abs_gt.shape[1] == pos_seq_first.shape[0]
                            else hist_abs_gt
                        )
                        gt_abs = gt_seq[..., :2]
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
                corrected_seq = self.cofe.infer_correction(
                    pos_seq_first,
                    hist_yaw_pred=None,
                    hist_resnet=None,
                    hist_seq_start_end=None,
                    hist_intent=intent_feature,
                )
                pos = corrected_seq.permute(1, 0, 2).contiguous()

        batch = pos.size(0)
        speed = torch.zeros_like(pos)
        speed[:, 1:] = pos[:, 1:] - pos[:, :-1]
        speed[:, 0] = speed[:, 1]

        hidden_size = self.args.hidden_size

        sloss, _, zsp, hsp, _ = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp = torch.mean(zsp, axis=1)

        ploss, _, zpo, hpo, _ = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo = torch.mean(zpo, axis=1)

        hpa = torch.zeros(batch, hidden_size, device=device)
        zpa = torch.zeros(batch, hidden_size, device=device)
        hsa = torch.zeros(batch, hidden_size, device=device)
        zsa = torch.zeros(batch, hidden_size, device=device)
        pb  = torch.zeros(batch, hidden_size, device=device)
        pbloss = torch.zeros(1, device=device)
        psloss = torch.zeros(1, device=device)

        if self.args.use_attribute:
            if ped_behavior is not None and hasattr(self, 'ped_behavior_encoder'):
                pbloss, _, zpa, hpa, _ = self.ped_behavior_encoder(ped_behavior)
                hpa = hpa[0].squeeze(0)
                zpa = torch.mean(zpa, axis=1)
            if scene_attribute is not None and hasattr(self, 'scene_attribute_encoder'):
                psloss, _, zsa, hsa, _ = self.scene_attribute_encoder(scene_attribute)
                hsa = hsa[0].squeeze(0)
                zsa = torch.mean(zsa, axis=1)
            if ped_attribute is not None and hasattr(self, 'mlp'):
                pb = self.mlp(ped_attribute)

        if lcf_features is not None and self.lcf_proj is not None:
            lcf_emb = self.lcf_proj(lcf_features)
            hpa = hpa + lcf_emb
            zpa = zpa + lcf_emb
        elif lcf_features is not None and self.lcf_proj is None:
            import warnings
            warnings.warn(
                "lcf_features provided but lcf_proj is None (lcf_feature_dim=0). "
                "Features will be discarded. Set lcf_feature_dim > 0 to enable LCF projection.",
                RuntimeWarning
            )

        himg = torch.zeros(batch, hidden_size, device=device)
        cimg = torch.zeros(batch, hidden_size, device=device)

        if self.args.use_image and images is not None and images.dim() == 5:
            B_img, T_img, C, H, W = images.size()
            _, (himg_raw, cimg_raw) = self.clstm(images)
            himg = self.pooling_h(himg_raw).view(himg_raw.size(0), -1)
            himg = self.linear_h(himg)
            cimg = self.pooling_c(cimg_raw).view(cimg_raw.size(0), -1)
            cimg = self.linear_c(cimg)

        himg_op = torch.zeros(batch, hidden_size, device=device)
        cimg_op = torch.zeros(batch, hidden_size, device=device)

        if optical is not None and self.args.use_opticalflow and hasattr(self, 'optical_resnet'):
            B_op, T_op, C_op, H_op, W_op = optical.size()
            optical_flat = optical.view(B_op * T_op, C_op, H_op, W_op)
            op_feats = self.optical_resnet(optical_flat)
            op_feats = op_feats.view(B_op, T_op, -1)
            _, (himg_op_h, cimg_op_h) = self.op_encoder(op_feats)
            himg_op = himg_op_h[-1].squeeze(0)
            cimg_op = cimg_op_h[-1].squeeze(0)

        intent_emb = torch.zeros(batch, hidden_size, device=device)
        if self.intent_proj is not None and intent_feature is not None:
            if intent_feature.dim() == 3:
                intent_prior = intent_feature.mean(dim=0)
            else:
                intent_prior = intent_feature
            intent_emb = self.intent_proj(intent_prior)
        elif intent_feature is not None and self.intent_proj is None:
            import warnings
            warnings.warn(
                "intent_feature provided but intent_proj is None (intent_feature_dim=0). "
                "Features will be discarded. Set intent_feature_dim > 0 to enable Intent projection.",
                RuntimeWarning
            )

        hds = hpo + hsp + hpa + hsa + pb + himg + himg_op + intent_emb
        zds = zpo + zsp + zpa + zsa + pb + cimg + cimg_op + intent_emb

        loss = ploss + sloss + pbloss + psloss + self.cofe_loss_weight * cofe_loss

        in_sp = speed[:, -1, :]
        speed_outputs = torch.tensor([], device=device)

        for i in range(self.args.output // self.args.skip):
            hds, zds = self.speed_decoder(in_sp, (hds, zds))
            speed_output = self.hardtanh(self.fc_speed(hds))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim=1)
            in_sp = speed_output.detach()

        if ego_idx is not None:
            speed_outputs = speed_outputs[ego_idx]

        return loss, cofe_loss, speed_outputs