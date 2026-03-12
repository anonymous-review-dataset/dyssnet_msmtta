import torch
import torch.nn as nn
import torch.nn.functional as F

class DyT(nn.Module):
    def __init__(self, channels, init_alpha=0.5):
        """
        Dynamic Tanh (DyT) layer as a replacement for BatchNorm2d

        Args:
            channels: Number of channels
            init_alpha: Initial value for the learnable scaling parameter
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Apply tanh with learnable scaling
        x = torch.tanh(self.alpha * x)

        # Apply channel-wise affine transformation (similar to normalization layers)
        # Reshape gamma and beta for proper broadcasting
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)

class SpatialShiftBlock(nn.Module):
    """
    SpatialShiftBlock splits the input channels into groups and shifts each group in a different direction.
    This operation is parameter‐free and helps exchange spatial context efficiently.
    """
    def __init__(self, shift_div=4):
        super().__init__()
        # shift_div determines into how many groups the channels are split.
        self.shift_div = shift_div


    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        fold = C // self.shift_div
        if fold == 0:
            return x  # not enough channels to shift in groups

        # Make a copy to apply shifts
        out = x.clone()

        # Split channels into four groups and roll each group in a different spatial directionD
        # Group 1: right shift (shift columns by +1)
        out[:, :fold, :, :] = torch.roll(x[:, :fold, :, :], shifts=1, dims=3)
        # Group 2: left shift (shift columns by -1)
        out[:, fold:2*fold, :, :] = torch.roll(x[:, fold:2*fold, :, :], shifts=-1, dims=3)
        # Group 3: down shift (shift rows by +1)
        out[:, 2*fold:3*fold, :, :] = torch.roll(x[:, 2*fold:3*fold, :, :], shifts=1, dims=2)
        # Group 4: up shift (shift rows by -1)
        out[:, 3*fold:4*fold, :, :] = torch.roll(x[:, 3*fold:4*fold, :, :], shifts=-1, dims=2)
        # Any remaining channels (if C is not divisible by shift_div) remain unchanged.
        return out

class DynamicSpatialShiftDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes, d_model=256, out_size=None, dropout=0, init_alpha=0.5):
        """
        A decoder that combines SegFormer's efficient MLP design with spatial shift operations,
        enhanced with Dynamic Tanh (DyT) instead of BatchNorm2d.

        Args:
            encoder_channels: List of encoder output channels [stage1, stage2, stage3, stage4]
            num_classes: Number of segmentation classes
            d_model: Internal feature dimension
            out_size: Final output size (H, W)
            dropout: Dropout rate
            init_alpha: Initial value for alpha in DyT layers
        """
        super().__init__()
        self.out_size = out_size

        # SegFormer-style linear projections with DyT instead of BatchNorm2d
        self.linear_c1 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], d_model, kernel_size=1, bias=False),
            DyT(d_model, init_alpha=init_alpha)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv2d(encoder_channels[1], d_model, kernel_size=1, bias=False),
            DyT(d_model, init_alpha=init_alpha)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(encoder_channels[2], d_model, kernel_size=1, bias=False),
            DyT(d_model, init_alpha=init_alpha)
        )
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(encoder_channels[3], d_model, kernel_size=1, bias=False),
            DyT(d_model, init_alpha=init_alpha)
        )

        # Spatial shift blocks for each stage
        self.spatial_shift1 = SpatialShiftBlock(shift_div=4)
        self.spatial_shift2 = SpatialShiftBlock(shift_div=4)
        self.spatial_shift3 = SpatialShiftBlock(shift_div=4)
        self.spatial_shift4 = SpatialShiftBlock(shift_div=4)

        # SegFormer-style MLP blocks for feature refinement
        self.mlp_c1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        self.mlp_c2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        self.mlp_c3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        self.mlp_c4 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

        # Feature fusion module with DyT
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            DyT(d_model, init_alpha=init_alpha),
            SpatialShiftBlock(shift_div=4),  # Add spatial shift to fused features
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.linear_pred = nn.Conv2d(d_model, num_classes, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        # Extract features from encoder
        c1, c2, c3, c4 = encoder_features

        # Apply linear projections and spatial shift
        n1 = self.linear_c1(c1)
        n1 = self.spatial_shift1(n1)
        n1 = self.mlp_c1(n1)

        n2 = self.linear_c2(c2)
        n2 = self.spatial_shift2(n2)
        n2 = self.mlp_c2(n2)

        n3 = self.linear_c3(c3)
        n3 = self.spatial_shift3(n3)
        n3 = self.mlp_c3(n3)

        n4 = self.linear_c4(c4)
        n4 = self.spatial_shift4(n4)
        n4 = self.mlp_c4(n4)

        # SegFormer-style multi-level feature aggregation
        # Upsample all features to the resolution of n1
        size = n1.shape[2:]

        n2 = F.interpolate(n2, size=size, mode='bilinear', align_corners=False)
        n3 = F.interpolate(n3, size=size, mode='bilinear', align_corners=False)
        n4 = F.interpolate(n4, size=size, mode='bilinear', align_corners=False)

        # Concatenate and fuse features
        x = torch.cat([n1, n2, n3, n4], dim=1)
        x = self.dropout(x)
        x = self.fusion(x)

        # Final prediction
        if self.out_size is not None:
            x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        x = self.linear_pred(x)

        return x