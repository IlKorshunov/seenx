import torch
import torch.nn as nn
import torch.nn.functional as F


class TransNetV2(nn.Module):
    def __init__(self, filters=16, L=3, S=2, D=1024, dropout_rate=0.5):
        super().__init__()
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(3, S, filters, stochastic_depth_drop_prob=0.0)] + [StackedDDCNNV2((filters * 2 ** (i - 1)) * 4, S, filters * 2**i) for i in range(1, L)]
        )
        feat_dim = sum((filters * 2**i) * 4 for i in range(L))
        self.frame_sim_layer = FrameSimilarity(feat_dim)
        self.color_hist_layer = ColorHistograms()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        out_dim = (filters * 2 ** (L - 1)) * 4 * 3 * 6 + 128 + 128
        self.fc1 = nn.Linear(out_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)
        self.eval()

    def forward(self, inputs):
        assert inputs.dtype == torch.uint8 and list(inputs.shape[2:]) == [27, 48, 3]
        x = inputs.permute(0, 4, 1, 2, 3).float().div_(255.0)
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], x.shape[2], -1)
        x = torch.cat([self.frame_sim_layer(block_features), x, self.color_hist_layer(inputs)], 2)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.cls_layer1(x), {"many_hot": self.cls_layer2(x)}


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters, stochastic_depth_drop_prob=0.0):
        super().__init__()
        self.DDCNN = nn.ModuleList([DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, activation=F.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)])
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x, shortcut = inputs, None
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
        x = F.relu(x)
        p = self.stochastic_depth_drop_prob
        x = (1 - p) * x + shortcut if p else x + shortcut
        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, activation=None):
        super().__init__()
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8)
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3)
        self.activation = activation

    def forward(self, inputs):
        x = torch.cat([self.Conv3D_1(inputs), self.Conv3D_2(inputs), self.Conv3D_4(inputs), self.Conv3D_8(inputs)], dim=1)
        x = self.bn(x)
        return self.activation(x) if self.activation else x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation, use_bias=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv3d(in_filters, 2 * filters, (1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.Conv3d(2 * filters, filters, (3, 1, 1), dilation=(dilation, 1, 1), padding=(dilation, 0, 0), bias=use_bias),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, sim_dim=128, window=101, out_dim=128):
        super().__init__()
        self.projection = nn.Linear(in_filters, sim_dim, bias=True)
        self.fc = nn.Linear(window, out_dim)
        self.lookup_window = window

    def forward(self, block_features):
        x = torch.cat([torch.mean(f, dim=[3, 4]) for f in block_features], dim=1)
        x = F.normalize(self.projection(x.transpose(1, 2)), p=2, dim=2)
        return _window_similarity_projection(x, self.lookup_window, self.fc)


class ColorHistograms(nn.Module):
    def __init__(self, window=101, out_dim=128):
        super().__init__()
        self.fc = nn.Linear(window, out_dim)
        self.lookup_window = window

    def forward(self, inputs):
        frames = inputs.int()
        B, T, H, W, _ = frames.shape
        flat = frames.view(B * T, H * W, 3)
        r, g, b = flat[:, :, 0] >> 5, flat[:, :, 1] >> 5, flat[:, :, 2] >> 5
        bins = (r << 6) + (g << 3) + b
        prefix = (torch.arange(B * T, device=frames.device) << 9).view(-1, 1)
        hist = torch.zeros(B * T * 512, dtype=torch.int32, device=frames.device)
        hist.scatter_add_(0, (bins + prefix).view(-1), torch.ones(bins.numel(), dtype=torch.int32, device=frames.device))
        x = F.normalize(hist.view(B, T, 512).float(), p=2, dim=2)
        return _window_similarity_projection(x, self.lookup_window, self.fc)


def _window_similarity_projection(x: torch.Tensor, lookup_window: int, fc: nn.Linear) -> torch.Tensor:
    batch_size, n_frames = x.shape[:2]
    sim = F.pad(torch.bmm(x, x.transpose(1, 2)), [(lookup_window - 1) // 2] * 2)
    window_idx = torch.arange(lookup_window, device=x.device).view(1, 1, -1) + torch.arange(n_frames, device=x.device).view(1, n_frames, 1)
    batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(batch_size, n_frames, lookup_window)
    time_idx = torch.arange(n_frames, device=x.device).view(1, n_frames, 1).expand(batch_size, n_frames, lookup_window)
    return F.relu(fc(sim[batch_idx, time_idx, window_idx.expand(batch_size, n_frames, lookup_window)]))
