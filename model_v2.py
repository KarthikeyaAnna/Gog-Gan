"""
Dog-GAN V2 (Unconditional): High-Capacity GAN Architecture

Key improvements over V1:
  1. Much wider channel schedule — minimum 64ch even at 256x256
  2. Spectral normalization in Generator for training stability
  3. Self-attention at TWO resolutions (16x16 and 32x32)
  4. Deeper residual blocks with skip connections
  5. Unconditional generation (no text embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Self-attention block for capturing long-range spatial dependencies (Discriminator version, supports double backward)."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Force attention to float32 to prevent FP16 PyTorch overflow/NaNs in torch.bmm
        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x.float()
            B, C, H, W = x_f32.shape
            q = self.query(x_f32).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C//8
            k = self.key(x_f32).view(B, -1, H * W)                      # B, C//8, HW
            v = self.value(x_f32).view(B, -1, H * W)                    # B, C, HW

            attn = torch.bmm(q, k)                                   # B, HW, HW
            attn = torch.softmax(attn / (C // 8) ** 0.5, dim=-1)    # scaled dot-product

            out = torch.bmm(v, attn.permute(0, 2, 1))               # B, C, HW
            out = out.view(B, C, H, W)

            res = self.gamma.float() * out + x_f32
        return res.to(x.dtype)


class SelfAttentionG(nn.Module):
    """Self-attention block for capturing long-range spatial dependencies (Generator version, optimized with FlashAttention)."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).permute(0, 2, 3, 1).reshape(B, 1, H * W, -1)   # B, 1, HW, C//8
        k = self.key(x).permute(0, 2, 3, 1).reshape(B, 1, H * W, -1)     # B, 1, HW, C//8
        v = self.value(x).permute(0, 2, 3, 1).reshape(B, 1, H * W, -1)   # B, 1, HW, C
        
        # F.scaled_dot_product_attention uses FlashAttention automatically if dtype is fp16/bf16
        out = F.scaled_dot_product_attention(q, k, v) # B, 1, HW, C
        
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2) # Reshape to B, C, H, W
        res = self.gamma * out + x
        return res


class GeneratorBlock(nn.Module):
    """Upsampling block for the generator with batch norm."""

    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=False)

        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity(),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        ) if in_ch != out_ch or upsample else nn.Identity()

    def forward(self, x):
        h = self.upsample(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)

        return self.act(h + self.shortcut(x))


class Generator(nn.Module):
    """
    Unconditional Generator V2.
    """

    CHANNEL_SCHEDULE = {
        0: 1.0,     # 4→8
        1: 0.5,     # 8→16
        2: 0.25,    # 16→32
        3: 0.125,   # 32→64
        4: 0.0625,  # 64→128
        5: 0.0625,  # 128→256
    }

    def __init__(self, image_size=256, noise_dim=128, base_ch=1024, min_ch=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.base_ch = base_ch
        self.min_ch = min_ch

        num_blocks = int(math.log2(image_size // 4))

        channels = []
        for i in range(num_blocks):
            mult_in = self.CHANNEL_SCHEDULE.get(i, 0.0625)
            mult_out = self.CHANNEL_SCHEDULE.get(i + 1, 0.0625)
            in_ch = max(min_ch, int(base_ch * mult_in)) if i > 0 else base_ch
            out_ch = max(min_ch, int(base_ch * mult_out))
            channels.append((in_ch, out_ch))

        # Project noise to spatial feature map (4x4)
        self.fc = nn.Linear(noise_dim, base_ch * 4 * 4)
        self.initial_bn = nn.BatchNorm2d(base_ch)
        self.act = nn.LeakyReLU(0.2, inplace=False)

        # Upsample blocks
        self.blocks = nn.ModuleList()
        for in_c, out_c in channels:
            self.blocks.append(GeneratorBlock(in_c, out_c))

        # Self-attention at 16x16 and 32x32
        self.attn_resolutions = {16, 32}
        self.attn_16 = SelfAttentionG(max(min_ch, int(base_ch * self.CHANNEL_SCHEDULE.get(2, 0.25))))
        self.attn_32 = SelfAttentionG(max(min_ch, int(base_ch * self.CHANNEL_SCHEDULE.get(3, 0.125))))

        # Final output layer
        final_ch = channels[-1][1]
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(final_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(final_ch, 3, 3, 1, 1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        z: (B, noise_dim) — random noise
        """
        x = self.fc(z)
        x = x.view(-1, self.base_ch, 4, 4)
        x = self.initial_bn(x)
        x = self.act(x)

        # Upsample through blocks
        for block in self.blocks:
            x = block(x)
            if x.shape[2] == 16:
                x = self.attn_16(x)
            elif x.shape[2] == 32:
                x = self.attn_32(x)

        return self.to_rgb(x)


class DiscriminatorBlock(nn.Module):
    """Downsampling block for the discriminator."""

    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=False),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.main = nn.Sequential(*layers)

        # Shortcut
        shortcut_layers = [nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))]
        if downsample:
            shortcut_layers.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class Discriminator(nn.Module):
    """
    Unconditional Discriminator V2 with spectral normalization.
    """

    def __init__(self, image_size=256, base_ch=64):
        super().__init__()
        self.image_size = image_size
        self.base_ch = base_ch

        num_blocks = int(math.log2(image_size // 4))
        channels = []
        out_ch = 512
        for i in range(num_blocks):
            in_ch = 3 if i == 0 else channels[-1][1]
            block_out_ch = max(64, out_ch // (2 ** (num_blocks - 1 - i)))
            channels.append((in_ch, block_out_ch))

        self.blocks = nn.ModuleList()
        for in_c, out_c in channels:
            self.blocks.append(DiscriminatorBlock(in_c, out_c))

        self.attn_resolutions = {}
        cur_res = image_size
        for i, (in_c, out_c) in enumerate(channels):
            cur_res = cur_res // 2
            if cur_res == 16:
                self.attn_resolutions[16] = out_c
            elif cur_res == 32:
                self.attn_resolutions[32] = out_c

        self.attn_modules = nn.ModuleDict()
        for res, ch in self.attn_resolutions.items():
            self.attn_modules[str(res)] = SelfAttention(ch)

        self.act = nn.LeakyReLU(0.2, inplace=False)

        # Unconditional score (is it a real image?)
        final_ch = channels[-1][1]
        self.fc_uncond = nn.utils.spectral_norm(nn.Linear(final_ch, 1))

    def forward(self, x):
        """
        x: (B, 3, image_size, image_size) — image
        """
        h = x
        for block in self.blocks:
            h = block(h)
            res = h.shape[2]
            if str(res) in self.attn_modules:
                h = self.attn_modules[str(res)](h)
        h = self.act(h)

        h_pooled = h.mean(dim=[2, 3])  # (B, final_ch)
        score_uncond = self.fc_uncond(h_pooled)
        return score_uncond

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(image_size=64).to(device)
    D = Discriminator(image_size=64).to(device)
    z = torch.randn(2, 128).to(device)
    fake = G(z)
    score = D(fake)
    print(f"Generator output shape: {fake.shape}")
    print(f"Discriminator output shape: {score.shape}")
