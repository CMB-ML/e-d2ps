import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Basic building blocks
# ----------------------------

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, norm_kind, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = norm_layer(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def norm_layer(ch, kind="bn"):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    elif kind == "gn":
        return nn.GroupNorm(num_groups=8, num_channels=ch)
    elif kind == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type {kind}")


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style block.
    """
    def __init__(self, in_ch, out_ch, norm_kind, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_ch, hidden_dim, norm_kind, kernel_size=1, stride=1))

        layers.extend([
            ConvNormAct(hidden_dim, hidden_dim, norm_kind, kernel_size=3, stride=stride, groups=hidden_dim),
            ConvNormAct(hidden_dim, out_ch, norm_kind, kernel_size=1, stride=1, act=False),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# MobileViT block
# ----------------------------

class MobileViTBlock(nn.Module):
    """
    Simplified MobileViT-style block:
      local convs -> unfold to patch tokens -> transformer -> fold back -> fusion
    """
    def __init__(
        self,
        in_ch,
        transformer_dim,
        norm_kind,
        depth=2,
        patch_size=(2, 2),
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()
        self.ph, self.pw = patch_size
        self.in_ch = in_ch
        self.transformer_dim = transformer_dim

        # Local representation
        self.local_rep = nn.Sequential(
            ConvNormAct(in_ch, in_ch, norm_kind, kernel_size=3, stride=1),
            ConvNormAct(in_ch, transformer_dim, norm_kind, kernel_size=1, stride=1),
        )

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(
                dim=transformer_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Project back and fuse
        self.project_back = ConvNormAct(transformer_dim, in_ch, norm_kind, kernel_size=1, stride=1)
        self.fuse = ConvNormAct(2 * in_ch, in_ch, norm_kind, kernel_size=3, stride=1)

    def _pad_if_needed(self, x):
        _, _, h, w = x.shape
        pad_h = (self.ph - h % self.ph) % self.ph
        pad_w = (self.pw - w % self.pw) % self.pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, h, w

    def forward(self, x):
        residual = x

        # Local conv features
        x = self.local_rep(x)  # (B, D, H, W)
        x, h0, w0 = self._pad_if_needed(x)
        b, d, h, w = x.shape

        # Unfold into non-overlapping patches
        # After unfold:
        #   x_unf shape: (B, D * ph * pw, N_patches)
        x_unf = F.unfold(x, kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))

        # Reformat to tokens:
        #   (B, N_patches, D * ph * pw)
        n_patches = x_unf.shape[-1]
        x_tokens = x_unf.transpose(1, 2)

        # Average within each patch position so token dim stays = D
        # Reshape: (B, N, D, ph*pw) -> mean over patch elements
        x_tokens = x_tokens.view(b, n_patches, d, self.ph * self.pw).mean(dim=-1)

        # Transformer
        x_tokens = self.transformer(x_tokens)  # (B, N, D)

        # Broadcast token back over each patch element
        x_tokens = x_tokens.unsqueeze(-1).expand(-1, -1, -1, self.ph * self.pw)
        x_tokens = x_tokens.reshape(b, n_patches, d * self.ph * self.pw).transpose(1, 2)

        # Fold back to feature map
        x = F.fold(
            x_tokens,
            output_size=(h, w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )

        # Undo padding
        x = x[:, :, :h0, :w0]

        # Project back + fuse with residual
        x = self.project_back(x)
        x = self.fuse(torch.cat([residual, x], dim=1))
        return x


# ----------------------------
# Full regressor
# ----------------------------

class MobileViTCMBRegressor(nn.Module):
    def __init__(
        self,
        nside,
        in_channels,
        n_outputs,
        norm_kind,
        widths=(32, 64, 96, 128),
        transformer_dims=(96, 120, 144),
        patch_size=(2, 2),
        transformer_depths=(2, 2, 2),
        num_heads=(4, 4, 4),
        mlp_ratio=2.0,
        dropout=0.0,
        head_hidden=512,
    ):
        super().__init__()

        self.nside = nside
        self.example_in_shape = (in_channels, 4*nside, 3*nside)

        w1, w2, w3, w4 = widths
        t1, t2, t3 = transformer_dims
        d1, d2, d3 = transformer_depths
        h1, h2, h3 = num_heads

        # Stem
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, w1, norm_kind, kernel_size=3, stride=2),  # 3N x 4N -> 1.5N x 2N
            InvertedResidual(w1, w1, norm_kind, stride=1, expand_ratio=2),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            InvertedResidual(w1, w2, norm_kind, stride=2, expand_ratio=4),
            MobileViTBlock(
                in_ch=w2,
                transformer_dim=t1,
                norm_kind=norm_kind,
                depth=d1,
                patch_size=patch_size,
                num_heads=h1,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            InvertedResidual(w2, w3, norm_kind, stride=2, expand_ratio=4),
            MobileViTBlock(
                in_ch=w3,
                transformer_dim=t2,
                norm_kind=norm_kind,
                depth=d2,
                patch_size=patch_size,
                num_heads=h2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            InvertedResidual(w3, w4, norm_kind, stride=1, expand_ratio=4),  # keep one stage from downsampling
            MobileViTBlock(
                in_ch=w4,
                transformer_dim=t3,
                norm_kind=norm_kind,
                depth=d3,
                patch_size=patch_size,
                num_heads=h3,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ),
        )

        # Final projection
        self.pre_head = nn.Sequential(
            ConvNormAct(w4, w4, norm_kind, kernel_size=1, stride=1),
            ConvNormAct(w4, w4, norm_kind, kernel_size=3, stride=1),
        )

        # Regression head
        # self.head = nn.Sequential(
        #     nn.Linear(w4, head_hidden),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(head_hidden, head_hidden),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(head_hidden, n_outputs),
        # )
        self.head = nn.Sequential(
            nn.Linear(2 * w4, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_outputs),
        )

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pre_head(x)

        # global average pooling near the end only
        # x = x.mean(dim=(2, 3))  # (B, C)

        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False)
        x = torch.cat([mean, std], dim=1)
        
        x = self.head(x)
        return x