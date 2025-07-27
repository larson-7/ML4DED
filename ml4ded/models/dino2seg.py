import math
import os
from collections import deque

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml4ded.dinov2.dinov2 import DINOv2
from ml4ded.models.blocks import FeatureFusionBlock, _make_scratch
import torch.nn.init as init

class TemporalExtractor(nn.Module):
    def __init__(self, embed_dim, num_temporal_tokens=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_temporal_tokens = num_temporal_tokens

        self.temporal_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1, groups=embed_dim  # depthwise conv
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(num_temporal_tokens)

    def forward(self, features):
        """
        Args:
            features: (B, N_spatial, C) — features from a single image
        Returns:
            temporal_tokens: (B, num_temporal_tokens, C)
        """
        B, N, C = features.shape

        x = features.transpose(1, 2)  # (B, C, N)
        x = self.temporal_conv(x)     # (B, C, N)
        x = F.relu(x)

        if x.device.type == "mps":
            x = x.cpu()
            x = self.temporal_pool(x)
            x = x.to("mps")
        else:
            x = self.temporal_pool(x)     # (B, C, num_tokens)
        temporal_tokens = x.transpose(1, 2)  # (B, num_tokens, C)

        return temporal_tokens



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

    def forward(self, query, key, value, need_weights=False):
        attn_out, attn_weights = self.cross_attn(query, key, value, need_weights=need_weights)
        return attn_out, attn_weights


class DPTSegmentationHead(nn.Module):
    def __init__(
            self,
            in_channels=768,  # transformer embedding dim
            features=256,  # decoder channels
            out_channels=[256, 512, 1024, 1024],  # for the 4 scale projections
            num_classes=6,
            use_bn=False,
            use_clstoken=False,
            use_temporal_consistency=False,
            num_temporal_tokens=2,
            cross_attn_heads=4,
            temporal_window=4,
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.use_temporal_consistency = use_temporal_consistency
        self.num_temporal_tokens = num_temporal_tokens
        self.temporal_window = temporal_window

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1)
            for out_ch in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                ) for _ in range(4)
            ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)

        head_features_1 = features
        head_features_2 = 32

        # reduce channels
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2,
            kernel_size=3, stride=1, padding=1
        )

        # final segmentation prediction
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0),
        )

        # Cross-attention for temporal consistency
        if use_temporal_consistency:
            # kernel size is dependent on how many tokens each extractor should extract
            self.temporal_extractor = TemporalExtractor(embed_dim=in_channels,
                                                        num_temporal_tokens=num_temporal_tokens,)

            self.cross_attn_block = CrossAttentionBlock(embed_dim=in_channels, num_heads=cross_attn_heads)
            self.gate = nn.Parameter(torch.tensor(1.0))


    def forward(self, out_features, patch_h, patch_w, previous_temporal_tokens=None, ):
        out = []
        temporal_tokens_out = None
        attn_weights = None

        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
            else:
                x = x[0]
                cls_token = None

            # Extract spatial tokens from last feature map layer
            if self.use_temporal_consistency and i == len(out_features) - 1:
                B, N, C = x.shape
                current_temporal_tokens = self.temporal_extractor(x)
                patch_tokens = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
                token_list = [patch_tokens, current_temporal_tokens]
                if self.use_clstoken and cls_token is not None:
                    token_list.insert(0, cls_token.unsqueeze(1))  # (B, 1, C)
                query = torch.cat(token_list, dim=1)  # (B, N_query, C)

                if previous_temporal_tokens is not None:
                    # previous_temporal_tokens: (B, N_temporal, C)
                    # Use as keys and values
                    attended, attn_weights = self.cross_attn_block(
                        query=query,
                        key=previous_temporal_tokens,
                        value=previous_temporal_tokens,
                        need_weights=True,
                    )  # (B, N_query, C)

                    # fuse gated attended output with original query (residual)
                    query = query + torch.sigmoid(self.gate) * attended

                # Extract tokens
                N_cls = 1 if self.use_clstoken and cls_token is not None else 0

                class_token_out = None
                if N_cls == 1:
                    class_token_out = query[:, 0, :]  # (B, C)
                spatial_tokens_out = query[:, N_cls:N_cls + N, :]  # (B, N, C)
                temporal_tokens_out = query[:, N_cls + N:, :]  # (B, N_spatial, C)
                temporal_tokens_out = temporal_tokens_out.permute(0, 2, 1)  # (B, C, N_spatial)

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

        return out, temporal_tokens_out, attn_weights

class Dino2Seg(nn.Module):
    def __init__(
            self,
            encoder='vitb',
            num_classes=40,
            image_height=476,
            image_width=630,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
            use_temporal_consistency=False,
            num_temporal_tokens=2,
            temporal_window=4,
            cross_attn_heads=4,
            model_weights_dir="",
            device="cuda",
    ):
        super(Dino2Seg, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        self.image_height = image_height
        self.image_width = image_width

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.device = device
        self.use_clstoken = use_clstoken

        # ──────────────── WEIGHT LOADING LOGIC ──────────────── #
        vitb_weight_file = None
        seg_weight_file = None

        if model_weights_dir and os.path.isdir(model_weights_dir):
            files = os.listdir(model_weights_dir)
            # Find backbone
            for f in files:
                if "vitb" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    vitb_weight_file = os.path.join(model_weights_dir, f)
            # Find segmentation head weights
            if use_temporal_consistency:
                # Strict: look for _temporal.pth or _temporal.pt at the end
                for f in files:
                    if f.endswith("_temporal.pth") or f.endswith("_temporal.pt"):
                        seg_weight_file = os.path.join(model_weights_dir, f)
                        break
            else:
                # Any seg*.pth/pt (ignore _temporal)
                for f in files:
                    if "seg" in f and (f.endswith(".pth") or f.endswith(".pt")) and not f.endswith("_temporal.pth") and not f.endswith("_temporal.pt"):
                        seg_weight_file = os.path.join(model_weights_dir, f)
                        break

        # Load backbone weights if available
        if vitb_weight_file:
            print(f"Loading ViT-b backbone weights from: {vitb_weight_file}")
            state_dict = torch.load(vitb_weight_file, map_location='cpu')
            missing_keys, unexpected_keys = self.pretrained.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print("[Backbone] Missing keys:", missing_keys)
            if unexpected_keys:
                print("[Backbone] Unexpected keys:", unexpected_keys)
            self.pretrained.eval()
            for param in self.pretrained.parameters():
                param.requires_grad = False
        else:
            print("No ViT-b backbone weights found.")

        # Setup segmentation head
        self.seg_head = DPTSegmentationHead(
            in_channels=features,
            num_classes=num_classes,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            use_temporal_consistency=use_temporal_consistency,
            num_temporal_tokens=num_temporal_tokens,
            cross_attn_heads=cross_attn_heads,
            temporal_window=temporal_window,
        )

        if seg_weight_file:
            print(f"Loading segmentation head weights from: {seg_weight_file}")
            state_dict = torch.load(seg_weight_file, map_location='cpu')
            missing_keys, unexpected_keys = self.seg_head.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print("[SegHead] Missing keys:", missing_keys)
                # Reinitialize only submodules that have missing weights
                modules_to_init = set(k.split('.')[0] for k in missing_keys)
                for name, module in self.seg_head.named_children():
                    if name in modules_to_init:
                        print(f"Reinitializing module: seg_head.{name}")
                        self.initialize_module_weights(module)

            if unexpected_keys:
                print("[SegHead] Unexpected keys:", unexpected_keys)
        else:
            if use_temporal_consistency:
                print("No temporal segmentation head weights found (no *_temporal.pth/pt file).")
            else:
                print("No segmentation head weights found.")

        if use_temporal_consistency:
            self.temporal_token_buffer = deque(maxlen=temporal_window)

        self.pretrained.to(self.device)
        self.seg_head.to(self.device)

    def initialize_module_weights(self, module):
        """
        Reinitialize weights in a module according to layer type.
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset_temporal_buffer(self):
        if hasattr(self, "temporal_token_buffer"):
            self.temporal_token_buffer.clear()

    def forward(self, x, previous_temporal_tokens=None):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)

        seg_logits, temporal_tokens, attn_weights = self.seg_head(
            out_features=features,
            patch_h=patch_h,
            patch_w=patch_w,
            previous_temporal_tokens=previous_temporal_tokens,
        )

        return seg_logits.squeeze(1), temporal_tokens, attn_weights

    @torch.no_grad()
    def infer_image(self, image: torch.Tensor):
        """
        Args:
            image: (1, 3, H, W) tensor
        Returns:
            seg_probs: (1, C, H, W)
            segmentation_pred: (1, H, W) numpy array
        """
        previous_temporal_tokens = None
        if self.seg_head.use_temporal_consistency and len(self.temporal_token_buffer) > 0:
            # Stack into (1, T * num_tokens, C)
            buffer_tokens = torch.cat(list(self.temporal_token_buffer), dim=1)  # [(1, num_tokens, C), ...]
            previous_temporal_tokens = buffer_tokens

        seg_logits, temporal_tokens, _ = self.forward(image, previous_temporal_tokens)

        # Update the buffer with the current frame's tokens
        if self.seg_head.use_temporal_consistency and temporal_tokens is not None:
            # (1, C, num_tokens) → (1, num_tokens, C)
            self.temporal_token_buffer.append(temporal_tokens.permute(0, 2, 1).detach().cpu().to(self.device))

        seg_probs = F.softmax(seg_logits, dim=1)
        segmentation_pred = torch.argmax(seg_probs, dim=1)

        return seg_probs, segmentation_pred.cpu().numpy()

    @torch.no_grad()
    def get_previous_temporal_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal tokens from the last N frames.

        Args:
            images: Tensor of shape (T, B, 3, H, W), where T = number of previous frames.

        Returns:
            Tensor of shape (B, T * num_temporal_tokens, C)
        """
        T, B, _, H, W = images.shape
        device = self.device
        images = images.to(device)

        all_temporal_tokens = []

        for t in range(T):
            frame = images[t]  # (B, 3, H, W)

            # Get last layer output from ViT encoder (returns [(tokens, cls_token)])
            out_features = self.pretrained.get_intermediate_layers(
                frame,
                [self.intermediate_layer_idx[self.encoder][-1]],  # last layer only
                return_class_token=True
            )  # returns List[(tokens, cls_token)]

            # Extract temporal tokens using seg_head
            temporal_tokens = self.seg_head.temporal_extractor(out_features[0][0])
            all_temporal_tokens.append(temporal_tokens)  # append (B, num_tokens, C)

        # Stack and flatten across T time steps
        temporal_token_stack = torch.stack(all_temporal_tokens, dim=1)  # (B, T, num_tokens, C)
        B, T, N_tokens, C = temporal_token_stack.shape
        temporal_token_stack = temporal_token_stack.view(B, T * N_tokens, C)  # (B, T*num_tokens, C)

        return temporal_token_stack


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
    dpt_head = DPTSegmentationHead(
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
