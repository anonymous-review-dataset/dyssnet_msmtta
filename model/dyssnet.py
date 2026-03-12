import torch
import torch.nn as nn
import torch.nn.functional as F
from DySS_decoder import DynamicSpatialShiftDecoder
from swin_umambad import get_swin_umamba_d_from_plans

class RetinexLaplacianBoundaryModule(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()

        # Illumination estimation
        self.illumination_branch = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]).float().reshape(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian_kernel)

        reduced_channels = max(4, channels // reduction)

        # Efficient channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Feature refinement
        self.feature_refine = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x, input_image):
        if input_image.shape[2:] != x.shape[2:]:
            input_image = F.interpolate(input_image, size=x.shape[2:], mode='bilinear')

        # Extract illumination
        illumination = self.illumination_branch(input_image)

        # Extract boundaries using Laplacian
        gray = input_image.mean(dim=1, keepdim=True)
        boundaries = torch.abs(F.conv2d(gray, self.laplacian, padding=1))
        boundaries = torch.sigmoid(boundaries * 5)  # Scale and normalize

        # Apply channel attention
        channel_att = self.channel_gate(x)

        # Combine illumination and boundary information
        combined_att = illumination * (1 + boundaries)
        enhanced = self.feature_refine(x * combined_att)

        return x * channel_att + enhanced +x


class DySSNet(nn.Module):
    def __init__(self,
                 num_classes=9,
                 pretrained=True,
                 d_model=128,
                 dropout=0):
        """
        Enhanced disease segmentation model with Retinex illumination module.

        Args:
            num_classes: Number of segmentation classes
            pretrained: Whether to use pretrained encoder weights
            d_model: Base feature dimension for decoder
        """
        super().__init__()

        # Main backbone (Swin-UMamba)
        mamba = get_swin_umamba_d_from_plans(num_input_channels=3, num_classes=9, use_pretrain=pretrained)
        self.encoder = mamba.vssm_encoder

        # Original feature dimensions
        feature_dims = [96, 192, 384, 768]

        # Create illumination modules for encoder features
        self.encoder_illum_modules = nn.ModuleList([
            RetinexLaplacianBoundaryModule(feature_dims[i])
            for i in range(4)
        ])


        # Decoder
        self.decoder = DynamicSpatialShiftDecoder(
            encoder_channels=feature_dims,
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
            out_size=224
        )

    def forward(self, x):
        # Extract features from main backbone
        main_features = self.encoder(x)

        # Apply illumination modules to encoder features
        features_for_decoder = []
        for i in range(4):
            # Resize input image to match feature map size
            resized_input = F.interpolate(
                x,
                size=main_features[i + 1].shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # Apply illumination module
            enhanced_feature = self.encoder_illum_modules[i](main_features[i + 1], resized_input)
            features_for_decoder.append(enhanced_feature)

        # Pass processed features to decoder
        logits = self.decoder(features_for_decoder)

        # Ensure output size matches input
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits,
                size=(x.shape[2], x.shape[3]),
                mode='bilinear',
                align_corners=True
            )

        return logits

if __name__ == "__main__":
    model = DySSNet(num_classes=9).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    out = model(x)
    print(out.shape)