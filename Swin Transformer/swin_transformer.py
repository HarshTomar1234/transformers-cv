"""
Swin Transformer -- From-Scratch Implementation
=================================================
4-stage Swin Transformer for image classification.
Patch Embed -> Stage 1-4 (SwinBlocks + PatchMerging) -> Classification Head

Reference:
    "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    Liu et al., ICCV 2021
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Window Partition / Reverse
# ---------------------------------------------------------------------------

def window_partition(x, win_size):
    """
    Partition a feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C) -- spatial feature map
        win_size: window size M

    Returns:
        windows: (B * num_windows, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, win_size, win_size, C)
    return x


def window_reverse(windows, win_size, H, W):
    """
    Reverse of window_partition -- merge windows back into full feature map.

    Args:
        windows: (B * num_windows, M*M, C) or (B * num_windows, M, M, C)
        win_size: window size M
        H, W: spatial dimensions of the original feature map

    Returns:
        x: (B, H, W, C)
    """
    num_windows_h = H // win_size
    num_windows_w = W // win_size
    B = int(windows.shape[0] // (num_windows_h * num_windows_w))

    x = windows.view(B, num_windows_h, num_windows_w, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
#  Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Multi-head self-attention within a local window, with relative position bias.

    The attention computation:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d) + B) V

    where B is the learnable relative position bias indexed by pairwise
    relative positions of tokens within the window.
    """

    def __init__(self, dim, num_heads, win_size, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win_size = win_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- relative position bias table ---
        # Each pair of tokens (i, j) inside a window has a relative position
        # (delta_row, delta_col) in range [-(M-1), +(M-1)].
        # We map this 2D offset to a 1D index and use it to look up a
        # learnable bias per head.

        coords = torch.stack(torch.meshgrid(
            torch.arange(win_size), torch.arange(win_size), indexing='ij'
        ))  # (2, M, M)
        coords_flat = coords.flatten(1)  # (2, M*M)

        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, M*M, M*M)
        rel = rel.permute(1, 2, 0).contiguous()                  # (M*M, M*M, 2)

        rel[:, :, 0] += win_size - 1
        rel[:, :, 1] += win_size - 1
        rel[:, :, 0] *= 2 * win_size - 1
        rel_index = rel.sum(-1)

        self.register_buffer('relative_position_index', rel_index)

        num_relative_positions = (2 * win_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B_, N, C)  where B_ = batch * num_windows, N = M*M
            mask: (num_windows, N, N) or None

        Returns:
            (B_, N, C)
        """
        B_, N, C = x.shape
        h = self.num_heads
        d = self.head_dim

        qkv = self.qkv(x).reshape(B_, N, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, h, N, d)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B_, h, N, N)

        # add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)  # (h, N, N)
        attn = attn + bias.unsqueeze(0)

        # apply mask for shifted-window attention
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, h, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, h, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
#  Swin Block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """
    A single Swin Transformer block:

        x -> LN -> (Shifted) Window Attention + skip
          -> LN -> MLP + skip

    If shift > 0, applies cyclic shift + attention mask (SW-MSA).
    If shift == 0, plain window attention (W-MSA).

    Swin blocks are used in pairs: one W-MSA followed by one SW-MSA,
    so that shifted windows create cross-window connections.
    """

    def __init__(self, dim, resolution, win_size, shift, num_heads,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.resolution = resolution
        self.win_size = win_size
        # disable shift when the window already spans the full spatial dim
        self.shift = shift if win_size < resolution[0] else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads, win_size, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

        H, W = resolution
        if self.shift > 0:
            self.register_buffer('attn_mask', self._build_mask(H, W))
        else:
            self.attn_mask = None

    def _build_mask(self, H, W):
        """
        Build the attention mask for shifted-window self-attention.

        After a cyclic shift, some windows contain tokens from different
        spatial regions. The mask ensures tokens from different regions
        do not attend to each other (blocked with -100).
        """
        ws = self.win_size
        s = self.shift
        img_mask = torch.zeros(1, H, W, 1)

        region_id = 0
        for h_slice in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            for w_slice in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                img_mask[:, h_slice, w_slice, :] = region_id
                region_id += 1

        windows = window_partition(img_mask, ws)            # (nW, M, M, 1)
        windows = windows.view(-1, ws * ws)                 # (nW, M*M)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)  # (nW, M*M, M*M)
        mask = mask.masked_fill(mask != 0, -100.0)
        return mask

    def forward(self, x):
        """
        Args:
            x: (B, L, C)  where L = H * W
        Returns:
            x: (B, L, C)
        """
        H, W = self.resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # partition -> attention -> merge
        x_win = window_partition(x, self.win_size)
        x_win = x_win.view(-1, self.win_size * self.win_size, C)

        attn_out = self.attn(x_win, mask=self.attn_mask)
        x = window_reverse(attn_out, self.win_size, H, W)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = shortcut + x.view(B, L, C)

        # MLP sub-block
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
#  Patch Merging (spatial downsampling between stages)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    Reduces spatial resolution by 2x and doubles the channel count.

    Gathers 2x2 neighboring patches, concatenates along channel dim (-> 4C),
    then projects down to 2C with a linear layer.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            x: (B, (H/2)*(W/2), 2C), new_H, new_W
        """
        B, _, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


# ---------------------------------------------------------------------------
#  Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Convert raw image into a sequence of patch tokens via strided convolution.
    """

    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H_img, W_img)
        Returns:
            x: (B, H*W, embed_dim), H, W
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
#  Swin Stage
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """
    One stage of the Swin Transformer: a stack of SwinBlocks that alternate
    between W-MSA (even index) and SW-MSA (odd index).
    """

    def __init__(self, dim, resolution, depth, num_heads, win_size,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                resolution=resolution,
                win_size=win_size,
                shift=0 if (i % 2 == 0) else win_size // 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
#  Swin Transformer (full model)
# ---------------------------------------------------------------------------

class SwinTransformer(nn.Module):
    """
    Full 4-stage Swin Transformer for image classification.

    Default configuration ("Swin-Pico") is sized for 32x32 inputs:
        patch_size  = 2   -> 16x16 token grid
        embed_dim   = 64
        depths      = [2, 2, 6, 2]
        num_heads   = [2, 4, 8, 16]
        window_size = 4

    The architecture scales as:
        Stage 1: 16x16,  64ch  (win=4)
        Stage 2:  8x8,  128ch  (win=4)
        Stage 3:  4x4,  256ch  (win=4)
        Stage 4:  2x2,  512ch  (win=2, no shift)
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        patch_size=2,
        embed_dim=64,
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        window_size=4,
        img_size=32,
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.,
    ):
        super().__init__()
        self.num_stages = len(depths)

        # patch embedding
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        grid = img_size // patch_size

        # stages + downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_stages):
            dim_i = embed_dim * (2 ** i)
            res_i = grid // (2 ** i)
            ws_i = min(window_size, res_i)

            self.stages.append(SwinStage(
                dim=dim_i,
                resolution=(res_i, res_i),
                depth=depths[i],
                num_heads=num_heads[i],
                win_size=ws_i,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            ))

            if i < self.num_stages - 1:
                self.downsamples.append(PatchMerging(dim_i))
            else:
                self.downsamples.append(nn.Identity())

        # classification head
        final_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(final_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, H, W = self.patch_embed(x)

        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i < self.num_stages - 1:
                x, H, W = self.downsamples[i](x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)      # global average pooling
        x = self.head_drop(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
#  Quick shape verification
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer().to(device)
    dummy = torch.randn(2, 3, 32, 32, device=device)
    out = model(dummy)

    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
