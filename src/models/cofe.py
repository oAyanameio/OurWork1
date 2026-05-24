import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout=0.3, layer_norm=False):
        super().__init__()
        self.dropout = dropout
        self.layer_norm = layer_norm

        feats = [in_size, *hidden_sizes, out_size]
        net = []
        net_norm = []
        for i in range(len(feats) - 1):
            net.append(nn.Linear(feats[i], feats[i + 1]))
            net_norm.append(nn.LayerNorm(feats[i + 1]))
        self.net = nn.ModuleList(net)
        self.net_norm = nn.ModuleList(net_norm) if layer_norm else None

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            if self.net_norm:
                x = self.net_norm[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.net[-1](x)
        return x


class CoFE(nn.Module):
    def __init__(self, input_size=2, hidden_size=96, num_layers=2,
                 use_resnet=False, no_abs=True, idxs=None,
                 use_intent=False, intent_dim=512,
                 dropout=0.3, layer_norm=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.no_abs = no_abs
        self.use_resnet = use_resnet
        self.use_intent = use_intent
        self.intent_dim = intent_dim
        self.idxs = idxs if idxs is not None else [6, 7]
        self.offset_idxs = torch.tensor(self.idxs, dtype=torch.long)
        self.feat_dim = len(self.idxs)

        self.f_offset = MLP(self.feat_dim, [hidden_size], hidden_size, dropout, layer_norm)

        if use_resnet:
            self.f_resnet = MLP(2048, [hidden_size], hidden_size, dropout, layer_norm)
        else:
            self.f_resnet = None

        if use_intent:
            self.f_intent = MLP(intent_dim, [hidden_size], hidden_size, dropout, layer_norm)
        else:
            self.f_intent = None

        corr_enc_in = hidden_size * 2
        if use_resnet:
            corr_enc_in += hidden_size
        if use_intent:
            corr_enc_in += hidden_size

        self.corr_enc = MLP(corr_enc_in, [hidden_size, 64], hidden_size, dropout, layer_norm)
        self.corr_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec_rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.corr_dec = MLP(hidden_size, [64, hidden_size], self.feat_dim, dropout, layer_norm)
        self.criterion = nn.MSELoss()

    def to(self, *args, **kwargs):
        self.offset_idxs = self.offset_idxs.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def ego_dists(hist_abs, seq_start_end):
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
        if self.f_resnet is None:
            return torch.empty((num_agents, 0), device=device)
        if hist_resnet is None:
            hist_resnet_t = torch.zeros(num_agents, 2048, device=device)
        else:
            hist_resnet_t = hist_resnet[t]
        return self.f_resnet(hist_resnet_t)

    def _encode_intent_step(self, hist_intent, t, num_agents, device):
        if self.f_intent is None:
            return torch.empty((num_agents, 0), device=device)
        if hist_intent is None:
            hist_intent_t = torch.zeros(num_agents, self.intent_dim, device=device)
        elif hist_intent.dim() == 2:
            hist_intent_t = hist_intent
        else:
            hist_intent_t = hist_intent[t]
        return self.f_intent(hist_intent_t)

    def train_correction(self, hist_abs_gt, hist_yaw_gt, hist_abs_pred,
                         hist_yaw_pred, hist_resnet, hist_seq_start_end,
                         hist_intent=None):
        timesteps, num_agents, _ = hist_abs_gt.shape
        device = hist_abs_gt.device
        MSE = torch.zeros(1).to(device)
        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)
        offset_gt = self.build_features(hist_abs_gt, hist_yaw_gt, hist_seq_start_end)
        offset_pred = self.build_features(hist_abs_pred, hist_yaw_pred, hist_seq_start_end)
        for t in range(timesteps):
            f_offset_t = self.f_offset(offset_pred[t])
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)
            f_intent_t = self._encode_intent_step(hist_intent, t, num_agents, device)
            x_enc = torch.cat([f_offset_t, f_resnet_t, f_intent_t, h[-1]], dim=-1)
            x_corr = self.corr_enc(x_enc)
            _, h = self.corr_rnn(x_corr.unsqueeze(0), h)
        for t in range(timesteps):
            x_dec = self.corr_dec(h[-1])
            MSE += torch.sqrt(self.criterion(x_dec, offset_gt[t]))
            x_dec_feat = self.f_offset(x_dec)
            _, h = self.corr_dec_rnn(x_dec_feat.unsqueeze(0), h)
        return MSE

    def infer_correction(self, hist_abs_pred, hist_yaw_pred=None,
                         hist_resnet=None, hist_seq_start_end=None,
                         hist_intent=None):
        timesteps, num_agents, _ = hist_abs_pred.shape
        device = hist_abs_pred.device
        rel_pred = torch.zeros_like(hist_abs_pred)
        rel_pred[1:] = hist_abs_pred[1:] - hist_abs_pred[:-1]
        if hist_yaw_pred is not None and hist_seq_start_end is not None:
            offset_pred = self.build_features(
                hist_abs_pred, hist_yaw_pred, hist_seq_start_end
            )
        else:
            offset_pred = rel_pred
        h = torch.zeros(self.num_layers, num_agents, self.hidden_size, device=device)
        for t in range(timesteps):
            f_offset_t = self.f_offset(offset_pred[t])
            f_resnet_t = self._encode_resnet_step(hist_resnet, t, num_agents, device)
            f_intent_t = self._encode_intent_step(hist_intent, t, num_agents, device)
            x_enc = torch.cat([f_offset_t, f_resnet_t, f_intent_t, h[-1]], dim=-1)
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