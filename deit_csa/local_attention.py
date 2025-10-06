
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.avg_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
        self.max_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        att = torch.sigmoid(self.avg_mlp(avg) + self.max_mlp(mx))
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=(3,5), dilations=(1,1), use_bn=False, beta_init=0.2, tau=1.5):
        super().__init__()
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(dilations, int):    dilations    = [dilations] * len(kernel_sizes)
        assert len(kernel_sizes) == len(dilations)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList() if use_bn else None
        for k, d in zip(kernel_sizes, dilations):
            pad = (k // 2) * d
            conv = nn.Conv2d(2, 1, k, padding=pad, dilation=d, bias=not use_bn)
            nn.init.zeros_(conv.weight)
            if conv.bias is not None: nn.init.zeros_(conv.bias)
            self.convs.append(conv)
            if use_bn: self.bns.append(nn.BatchNorm2d(1))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.tau  = float(tau)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx  = x.amax(1, keepdim=True)
        m   = torch.cat([avg, mx], dim=1)
        acc = 0.0
        for i, conv in enumerate(self.convs):
            y = conv(m)
            if self.bns is not None: y = self.bns[i](y)
            acc = acc + y
        acc = acc / len(self.convs)
        att = torch.sigmoid(acc / self.tau)
        return x * (1.0 + self.beta * (att - 0.5))

class ScaleGate(nn.Module):
    def __init__(self, dim, hidden=128, n_scales=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Conv2d(dim * n_scales, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_scales, 1, bias=True)
        )
    def forward(self, feats):
        g = torch.cat([self.pool(f) for f in feats], dim=1)
        logits = self.mlp(g)
        return torch.softmax(logits, dim=1)

class LocalBlock(nn.Module):
    def __init__(self, dim, use_multiscale=True, use_channel=True, use_spatial=True,
                 spatial_beta=0.2, spatial_tau=1.5, spatial_kernels=(3,5), spatial_dils=(1,1)):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_channel    = use_channel
        self.use_spatial    = use_spatial

        if use_multiscale:
            self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, dilation=1, groups=dim, bias=False)
            self.dw2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
            self.dw3 = nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False)
            self.scale_gate = ScaleGate(dim, hidden=128, n_scales=3)
        else:
            self.dw1 = self.dw2 = self.dw3 = self.scale_gate = None

        self.chan = ChannelAttention(dim) if use_channel else None
        self.spa  = SpatialAttention(kernel_sizes=spatial_kernels, dilations=spatial_dils,
                                     beta_init=spatial_beta, tau=spatial_tau) if use_spatial else None

        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn   = nn.BatchNorm2d(dim)
        self.act  = nn.GELU()
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.use_multiscale:
            f1, f2, f3 = self.dw1(x), self.dw2(x), self.dw3(x)
            w  = self.scale_gate([f1, f2, f3])
            f  = w[:,0:1]*f1 + w[:,1:2]*f2 + w[:,2:3]*f3
        else:
            f = x
        if self.use_channel: f = self.chan(f)
        if self.use_spatial: f = self.spa(f)
        f = self.fuse(f); f = self.bn(f); f = self.act(f)
        return x + self.res_scale * f
