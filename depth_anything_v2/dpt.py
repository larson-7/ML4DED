import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .util.blocks import FeatureFusionBlock, _make_scratch


class Token1DCNNExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, num_tokens_out=1, kernel_size=3):
        """
        Args:
            in_channels: C (feature dim)
            out_channels: output dim for each temporal token (can be = in_channels)
            num_tokens_out: number of output tokens (length after 1D CNN, usually 1)
            kernel_size: 1D conv kernel size
        """
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2
        )
        self.pool = nn.AdaptiveAvgPool1d(num_tokens_out)  # Downsample N -> num_tokens_out

    def forward(self, x):
        """
        x: (B, C, N)
        Returns: (B, out_channels, num_tokens_out)
        """
        x = self.conv1d(x)    # (B, out_channels, N)
        x = self.pool(x)      # (B, out_channels, num_tokens_out)
        return x


# --- Utility Functions ---
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_out, _ = self.cross_attn(query, key, value)
        return attn_out


# --- DPT Head ---

class DPTHead(nn.Module):
    def __init__(
            self,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[256, 512, 1024, 1024],
            use_clstoken=False,
            use_temporal_consistency=False,
            num_temporal_tokens=2,
            cross_attn_heads=4,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.use_temporal_consistency = use_temporal_consistency
        self.num_temporal_tokens = num_temporal_tokens

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        # Cross-attention for temporal consistency
        if use_temporal_consistency:
            # kernel size is dependent on how many tokens each extractor should extract
            self.temporal_extractor = nn.ModuleList([Token1DCNNExtractor(in_channels=in_channels,
                                                                         out_channels=in_channels, num_tokens_out=1,
                                                                         kernel_size=3) for x in
                                                     range(self.num_temporal_tokens)])
            self.cross_attn_block = CrossAttentionBlock(embed_dim=in_channels, num_heads=cross_attn_heads)

    def extract_temporal_tokens(self, x):
        B, N, C = x.shape
        assert N % self.num_temporal_tokens == 0, "Number of total tokens must be divisible by number of temporal tokens"
        tokens_per_extractor = N // self.num_temporal_tokens
        x_grouped = x.view(B, self.num_temporal_tokens, tokens_per_extractor,
                           C)  # (B, num_temporal_tokens, tokens_per_extractor, C)

        temporal_tokens = []
        for i, extractor in enumerate(self.temporal_extractor):
            group = x_grouped[:, i, :, :].transpose(1, 2)  # (B, C, tokens_per_extractor)
            token = extractor(group)  # (B, C_out, 1)
            temporal_tokens.append(token.squeeze(-1))  # (B, C_out)
        temporal_tokens = torch.stack(temporal_tokens, dim=1)  # (B, num_extractors, C_out)
        return temporal_tokens

    def forward(self, out_features, patch_h, patch_w, previous_temporal_tokens=None, ):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
            else:
                x = x[0]
                cls_token = None

            # Extract spatial tokens from last feature map layer
            if self.use_temporal_consistency and i == len(out_features) - 1:
                B, N, C = x.shape
                temporal_tokens = self.extract_temporal_tokens(x)
                patch_tokens = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
                token_list = [patch_tokens, temporal_tokens]
                if self.use_clstoken and cls_token is not None:
                    token_list.insert(0, cls_token.unsqueeze(1))  # (B, 1, C)
                query = torch.cat(token_list, dim=1)  # (B, N_query, C)

                if previous_temporal_tokens is not None:
                    # previous_temporal_tokens: (B, N_temporal, C)
                    # Use as keys and values
                    attended = self.cross_attn_block(
                        query=query,
                        key=previous_temporal_tokens,
                        value=previous_temporal_tokens
                    )  # (B, N_query, C)

                    # fuse attended output with original query (residual)
                    query = query + attended

                # Extract tokens
                N_cls = 1 if self.use_clstoken and cls_token is not None else 0

                class_token_out = None
                if N_cls == 1:
                    class_token_out = query[:, 0, :]  # (B, C)
                spatial_tokens_out = query[:, N_cls:N_cls + N, :]  # (B, N, C)
                temporal_tokens_out = query[:, N_cls + N:, :]  # (B, N_spatial, C)
                temporal_tokens_out = temporal_tokens_out.permute(0, 2, 1)  # (B, C, N_spatial)

                temporal_class_tokens_out = torch.cat((class_token_out.unsqueeze(-1), temporal_tokens_out), dim=-1)
                x = spatial_tokens_out

            else:
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            # reassemble blocks
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out  # (B, C, H, W)

        # --- Standard DPT Head fusion/decoding ---
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out, temporal_class_tokens_out


def main():
    B = 2  # batch size
    C = 256  # input channel size
    patch_h = patch_w = 12
    num_layers = 4
    out_channels = [256, 512, 1024, 1024]

    out_features = []
    N = patch_h * patch_w  # ALWAYS FIXED for ViT!
    for l in range(num_layers):
        tokens = torch.randn(B, N, C)
        cls_token = torch.randn(B, C)
        out_features.append((tokens, cls_token))

    # Fake previous_temporal_tokens: (B, N_temporal, C)
    N_temporal = 8
    previous_temporal_tokens = torch.randn(B, N_temporal, C)

    # Instantiate head
    dpt_head = DPTHead(
        in_channels=C,
        use_clstoken=True,
        use_temporal_consistency=True,
        num_temporal_tokens=N_temporal,
    )

    # Run head
    out, temporal_class_tokens_out = dpt_head(
        out_features,
        patch_h=patch_h,
        patch_w=patch_w,
        previous_temporal_tokens=previous_temporal_tokens,
    )

    print(f"Output dpt shape: {out.shape}")
    print(f"Temporal+class tokens shape: {temporal_class_tokens_out.shape}")


if __name__ == '__main__':
    main()
