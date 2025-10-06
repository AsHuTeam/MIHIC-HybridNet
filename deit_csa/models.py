
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .local_attention import LocalBlock

class HybridDeiTParallel(nn.Module):
    def __init__(self, num_classes=7, deit_name='deit_base_patch16_224', img_size=128, drop=0.1,
                 enable_local=True, use_multiscale=True, use_channel=True, use_spatial=True,
                 gate_mode='scalar', spatial_beta=0.2, spatial_tau=1.5,
                 spatial_kernels=(3,5), spatial_dils=(1,1), use_gem=False):
        super().__init__()
        self.vit = timm.create_model(deit_name, pretrained=True, num_classes=0, img_size=img_size)
        self.patch_embed = self.vit.patch_embed
        self.cls_token   = nn.Parameter(self.vit.cls_token.detach().clone())
        self.pos_embed   = nn.Parameter(self.vit.pos_embed.detach().clone())
        self.pos_drop    = self.vit.pos_drop
        self.blocks      = self.vit.blocks
        self.norm        = self.vit.norm
        self.embed_dim   = self.vit.embed_dim
        if hasattr(self.patch_embed, "img_size"):
            try: self.patch_embed.img_size = (img_size, img_size)
            except Exception: pass

        self.enable_local = enable_local
        if enable_local:
            self.local = LocalBlock(
                dim=self.embed_dim,
                use_multiscale=use_multiscale,
                use_channel=use_channel,
                use_spatial=use_spatial,
                spatial_beta=spatial_beta,
                spatial_tau=spatial_tau,
                spatial_kernels=spatial_kernels,
                spatial_dils=spatial_dils,
            )
            self.gate = nn.Parameter(torch.zeros(self.embed_dim) if gate_mode=='channel' else torch.zeros(1))
        else:
            self.local = None; self.gate = None
        self.gate_mode = gate_mode

        self.dropout   = nn.Dropout(drop)
        self.classifier = nn.Linear(self.embed_dim * 2, num_classes)

        n_tokens = self.pos_embed.shape[1] - 1
        self.orig_grid = int((n_tokens) ** 0.5)
        patch = self.patch_embed.patch_size[0] if isinstance(self.patch_embed.patch_size, tuple) else self.patch_embed.patch_size
        assert img_size % patch == 0
        self.grid = img_size // patch

    @torch.no_grad()
    def _interpolate_pos_embed(self, pos_embed, H, W):
        cls_pos  = pos_embed[:, :1, :]
        patch_pe = pos_embed[:, 1:, :]
        C = patch_pe.shape[-1]
        patch_pe = patch_pe.reshape(1, self.orig_grid, self.orig_grid, C).permute(0,3,1,2)
        patch_pe = F.interpolate(patch_pe, size=(H, W), mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0,2,3,1).reshape(1, H*W, C)
        return torch.cat([cls_pos, patch_pe], dim=1)

    def forward(self, x):
        B = x.size(0)
        tokens_or_map = self.patch_embed(x)
        if tokens_or_map.dim() == 3:
            B_, N, C = tokens_or_map.shape
            H = W = int(N ** 0.5); assert H * W == N
            tokens = tokens_or_map
            F_MCSA_in  = tokens.view(B, H, W, self.embed_dim).permute(0,3,1,2)
        else:
            B_, C, H, W = tokens_or_map.shape
            F_MCSA_in = tokens_or_map
            tokens = F_MCSA_in.flatten(2).transpose(1, 2)

        cls_tok = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tok, tokens], dim=1)
        seq = self.pos_drop(seq + self._interpolate_pos_embed(self.pos_embed, H, W))
        for blk in self.blocks: seq = blk(seq)
        seq = self.norm(seq)
        F_DeiT_cls  = seq[:, 0, :]
        F_DeiT = seq[:, 1:, :].view(B, H, W, self.embed_dim).permute(0,3,1,2)

        if self.enable_local:
            F_MCSA = self.local(F_MCSA_in)
            if self.gate_mode == 'channel':
                alpha = torch.sigmoid(self.gate)[None, :, None, None]
            else:
                alpha = torch.sigmoid(self.gate)
            fused = alpha * F_DeiT + (1.0 - alpha) * F_MCSA
        else:
            fused = F_DeiT

        u = fused.mean(dim=(2,3))
        feats = torch.cat([F_DeiT_cls, u], dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.ones(1) * p); self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.nn.functional.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class SpatialFusionGate(nn.Module):
    def __init__(self, c, shrink=4):
        super().__init__()
        hidden = max(c // shrink, 4)
        self.net = nn.Sequential(
            nn.Conv2d(2*c, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, F_DeiT, F_MCSA):
        g = self.net(torch.cat([F_DeiT, F_MCSA], dim=1))
        mixed = g * F_DeiT + (1.0 - g) * F_MCSA
        return mixed, g

class HybridDeiTParallelBoosted(nn.Module):
    def __init__(self, num_classes=7, deit_name='deit_base_patch16_224', img_size=224, drop=0.1):
        super().__init__()
        self.vit = timm.create_model(deit_name, pretrained=True, num_classes=0, img_size=img_size)
        self.patch_embed = self.vit.patch_embed
        self.cls_token   = nn.Parameter(self.vit.cls_token.detach().clone())
        self.pos_embed   = nn.Parameter(self.vit.pos_embed.detach().clone())
        self.pos_drop    = self.vit.pos_drop
        self.blocks      = self.vit.blocks
        self.norm        = self.vit.norm
        self.embed_dim   = self.vit.embed_dim
        if hasattr(self.patch_embed, "img_size"):
            try: self.patch_embed.img_size = (img_size, img_size)
            except: pass

        self.local = LocalBlock(self.embed_dim, use_multiscale=True, use_channel=True, use_spatial=True,
                                spatial_kernels=(3,5), spatial_dils=(1,1), spatial_beta=0.2, spatial_tau=1.5)
        self.gate_ch = nn.Parameter(torch.zeros(self.embed_dim))
        self.sfuse = SpatialFusionGate(self.embed_dim, shrink=4)
        self.gem = GeM(p=3.0)
        self.dropout   = nn.Dropout(drop)
        self.classifier = nn.Linear(self.embed_dim * 2, num_classes)

        n_tokens = self.pos_embed.shape[1] - 1
        self.orig_grid = int((n_tokens) ** 0.5)
        patch = self.patch_embed.patch_size[0] if isinstance(self.patch_embed.patch_size, tuple) else self.patch_embed.patch_size
        assert img_size % patch == 0
        self.grid = img_size // patch

    @torch.no_grad()
    def _interp_pos(self, pos_embed, H, W):
        cls_pos  = pos_embed[:, :1, :]
        patch_pe = pos_embed[:, 1:, :]
        C = patch_pe.shape[-1]
        patch_pe = patch_pe.reshape(1, self.orig_grid, self.orig_grid, C).permute(0,3,1,2)
        patch_pe = torch.nn.functional.interpolate(patch_pe, size=(H, W), mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0,2,3,1).reshape(1, H*W, C)
        return torch.cat([cls_pos, patch_pe], dim=1)

    def forward(self, x):
        B = x.size(0)
        tokens_or_map = self.patch_embed(x)
        if tokens_or_map.dim() == 3:
            B_, N, C = tokens_or_map.shape
            H = W = int(N ** 0.5); assert H*W == N
            tokens = tokens_or_map
            F_MCSA_in  = tokens.view(B, H, W, self.embed_dim).permute(0,3,1,2)
        else:
            B_, C, H, W = tokens_or_map.shape
            F_MCSA_in = tokens_or_map
            tokens = F_MCSA_in.flatten(2).transpose(1, 2)

        cls_tok = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tok, tokens], dim=1)
        seq = self.pos_drop(seq + self._interp_pos(self.pos_embed, H, W))
        for blk in self.blocks: seq = blk(seq)
        seq = self.norm(seq)
        F_DeiT_cls  = seq[:, 0, :]
        F_DeiT = seq[:, 1:, :].view(B, H, W, self.embed_dim).permute(0,3,1,2)

        F_MCSA = self.local(F_MCSA_in)
        P_mix, g = self.sfuse(F_DeiT, F_MCSA)
        alpha = torch.sigmoid(self.gate_ch)[None, :, None, None]
        fused = alpha * F_DeiT + (1.0 - alpha) * P_mix

        u = self.gem(fused)
        feats = torch.cat([F_DeiT_cls, u], dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)



def make_deit_plus_multiscale_channel_spatial_boosted(num_classes, deit_name='deit_base_patch16_224', img_size=224, drop=0.1, **kw):
    return HybridDeiTParallelBoosted(num_classes, deit_name=deit_name, img_size=img_size, drop=drop)
