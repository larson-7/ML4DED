import math
import os
from collections import deque

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Assuming these imports exist in your project structure
from ml4ded.dinov2.dinov2_transformer import DINOv2
from ml4ded.models.blocks import FeatureFusionBlock, _make_scratch


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


class TemporalExtractor(nn.Module):
    def __init__(self, embed_dim, num_temporal_tokens=16):
        super().__init__()
        # Learnable queries that "ask" for specific info from the image
        self.query_embed = nn.Parameter(torch.randn(1, num_temporal_tokens, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, features):
        # features: (B, N_spatial, C)
        B = features.shape[0]

        # Expand queries for batch
        queries = self.query_embed.expand(B, -1, -1)

        # Attention: Queries look at the Spatial Features to extract info
        # query=queries, key=features, value=features
        out, _ = self.attn(queries, features, features)

        return out  # (B, num_temporal_tokens, C)


# class TemporalExtractor(nn.Module):
#     def __init__(self, embed_dim, num_temporal_tokens=4):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_temporal_tokens = num_temporal_tokens

#         self.temporal_conv = nn.Conv1d(
#             embed_dim,
#             embed_dim,
#             kernel_size=3,
#             padding=1,
#             groups=embed_dim,  # depthwise conv
#         )

#         self.temporal_pool = nn.AdaptiveAvgPool1d(num_temporal_tokens)

#     def forward(self, features):
#         """
#         Args:
#             features: (B, N_spatial, C)
#         Returns:
#             temporal_tokens: (B, num_temporal_tokens, C)
#         """
#         B, N, C = features.shape

#         x = features.transpose(1, 2)  # (B, C, N)
#         x = self.temporal_conv(x)  # (B, C, N)
#         x = F.relu(x)

#         # MPS device check for adaptive pool compatibility
#         if x.device.type == "mps":
#             x = x.cpu()
#             x = self.temporal_pool(x)
#             x = x.to("mps")
#         else:
#             x = self.temporal_pool(x)  # (B, C, num_tokens)

#         temporal_tokens = x.transpose(1, 2)  # (B, num_tokens, C)

#         return temporal_tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, need_weights=False):
        attn_out, attn_weights = self.cross_attn(
            query, key, value, need_weights=need_weights
        )
        return attn_out, attn_weights


class DPTSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels=768,
        features=256,
        out_channels=[256, 512, 1024, 1024],
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
        self.in_channels = in_channels

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                    for _ in range(4)
                ]
            )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0),
        )

        if use_temporal_consistency:
            self.temporal_extractor = TemporalExtractor(
                embed_dim=in_channels,
                num_temporal_tokens=num_temporal_tokens,
            )
            self.cross_attn_block = CrossAttentionBlock(
                embed_dim=in_channels, num_heads=cross_attn_heads
            )
            self.gate = nn.Parameter(
                torch.tensor(0.0)
            )  # Initialize gate at 0 (identity) to start

            # --- NEW: Learnable Initial State for Cold Starts ---
            # Instead of passing None or Zeros, the model learns a "default" history.
            self.init_temporal_tokens = nn.Parameter(
                torch.randn(1, num_temporal_tokens, in_channels)
            )

    def forward(self, out_features, patch_h, patch_w, previous_temporal_tokens=None):
        out = []
        temporal_tokens_out = None
        attn_weights = None

        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
            else:
                x = x[0]
                cls_token = None

            # --- Temporal Logic applied at the LAST layer of features ---
            if self.use_temporal_consistency and i == len(out_features) - 1:
                B, N, C = x.shape

                # 1. Extract tokens from CURRENT frame to be passed to NEXT frame
                current_temporal_tokens = self.temporal_extractor(
                    x
                )  # (B, num_tokens, C)

                # 2. Prepare Query: [CLS, Patch_Tokens, Current_Temporal_Tokens]
                patch_tokens = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
                token_list = [patch_tokens, current_temporal_tokens]

                if self.use_clstoken and cls_token is not None:
                    token_list.insert(0, cls_token.unsqueeze(1))  # (B, 1, C)

                query = torch.cat(token_list, dim=1)  # (B, N_query, C)

                # 3. Handle Historical Context
                # If None (Cold Start), use learned init parameter
                if previous_temporal_tokens is None:
                    previous_temporal_tokens = self.init_temporal_tokens.expand(
                        B, -1, -1
                    )

                # Cross Attention: Query=Current, Key/Value=History
                attended, attn_weights = self.cross_attn_block(
                    query=query,
                    key=previous_temporal_tokens,
                    value=previous_temporal_tokens,
                    need_weights=True,
                )

                # Residual Connection with Gate
                query = query + torch.sigmoid(self.gate) * attended

                # 4. Unpack Tokens for decoding
                N_cls = 1 if self.use_clstoken and cls_token is not None else 0

                class_token_out = None
                if N_cls == 1:
                    class_token_out = query[:, 0, :]  # (B, C)

                spatial_tokens_out = query[:, N_cls : N_cls + N, :]  # (B, N, C)

                # These are the tokens we just extracted/refined to pass to the NEXT frame
                temporal_tokens_out = query[:, N_cls + N :, :]  # (B, num_temporal, C)

                # Note: We return temporal_tokens_out as (B, N, C) so it's ready for next CrossAttn
            else:
                if self.use_clstoken and cls_token is not None:
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))

            # Reshape tokens to spatial grid
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0], x.shape[1], patch_h, patch_w)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # --- Standard DPT Fusion ---
        layer_1_rn = self.scratch.refinenet1(layer_1)
        layer_2_rn = self.scratch.refinenet2(layer_2)
        layer_3_rn = self.scratch.refinenet3(layer_3)
        layer_4_rn = self.scratch.refinenet4(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out, temporal_tokens_out, attn_weights


class Dino2Seg(nn.Module):
    def __init__(
        self,
        encoder="vitb",
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
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.image_height = image_height
        self.image_width = image_width

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.device = device
        self.use_clstoken = use_clstoken

        # --- Weight Loading Logic ---
        vitb_weight_file = None
        seg_weight_file = None

        if model_weights_dir and os.path.isdir(model_weights_dir):
            files = os.listdir(model_weights_dir)
            for f in files:
                if "vitb" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    vitb_weight_file = os.path.join(model_weights_dir, f)

            # Smart search for seg head
            if use_temporal_consistency:
                for f in files:
                    if f.endswith("_temporal.pth") or f.endswith("_temporal.pt"):
                        seg_weight_file = os.path.join(model_weights_dir, f)
                        break
            else:
                for f in files:
                    if (
                        "seg" in f
                        and (f.endswith(".pth") or f.endswith(".pt"))
                        and not "temporal" in f
                    ):
                        seg_weight_file = os.path.join(model_weights_dir, f)
                        break

        if vitb_weight_file:
            print(f"Loading ViT-b backbone weights from: {vitb_weight_file}")
            state_dict = torch.load(vitb_weight_file, map_location="cpu")
            self.pretrained.load_state_dict(state_dict, strict=False)
            self.pretrained.eval()
            for param in self.pretrained.parameters():
                param.requires_grad = False
        else:
            print(
                "No ViT-b backbone weights found. (Ensure internet access for auto-download if needed)"
            )

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
            state_dict = torch.load(seg_weight_file, map_location="cpu")
            missing_keys, unexpected_keys = self.seg_head.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                print("[SegHead] Missing keys:", missing_keys)
        else:
            print("No segmentation head weights found. Initializing randomly.")

        if use_temporal_consistency:
            self.temporal_token_buffer = deque(maxlen=temporal_window)

        self.pretrained.to(self.device)
        self.seg_head.to(self.device)

    def reset_temporal_buffer(self):
        if hasattr(self, "temporal_token_buffer"):
            self.temporal_token_buffer.clear()

    def forward(self, x, previous_temporal_tokens=None):
        """
        Single frame forward pass.
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        seg_logits, temporal_tokens, attn_weights = self.seg_head(
            out_features=features,
            patch_h=patch_h,
            patch_w=patch_w,
            previous_temporal_tokens=previous_temporal_tokens,
        )

        return seg_logits.squeeze(1), temporal_tokens, attn_weights

    def forward_clip(self, clip_tensor):
        """
        Efficient forward pass for TRAINING on video clips.

        Args:
            clip_tensor: (B, T, 3, H, W)

        Returns:
            clip_logits: (B, T, NumClasses, H, W)
        """
        B, T, C, H, W = clip_tensor.shape

        # 1. Run Backbone on all frames at once (Efficient)
        # Flatten: (B*T, 3, H, W)
        flat_images = clip_tensor.view(B * T, C, H, W)

        with torch.no_grad():
            flat_features = self.pretrained.get_intermediate_layers(
                flat_images,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True,
            )

        # 2. Re-structure features for temporal loop
        # We need a list of length T, where each element is the standard DINO output structure for Batch B
        sequence_features = []
        for t in range(T):
            frame_feats_t = []
            for layer_idx in range(len(flat_features)):
                p_tokens, c_token = flat_features[layer_idx]

                # 1. Handle Patch Tokens: (B*T, N, D) -> (B, T, N, D)
                N, D = p_tokens.shape[1], p_tokens.shape[2]
                p_t = p_tokens.view(B, T, N, D)[:, t, :, :]

                # 2. Handle Class Token: (B*T, D) -> (B, T, D)
                c_t = c_token.view(B, T, D)[:, t, :]

                frame_feats_t.append((p_t, c_t))
            sequence_features.append(frame_feats_t)

        # 3. Unroll Temporal Loop
        logits_list = []
        previous_temporal_tokens = (
            None  # Start with Cold Start (learnable init handled in Head)
        )

        patch_h, patch_w = H // 14, W // 14

        for t in range(T):
            seg_logits, next_temporal_tokens, _ = self.seg_head(
                out_features=sequence_features[t],
                patch_h=patch_h,
                patch_w=patch_w,
                previous_temporal_tokens=previous_temporal_tokens,
            )

            logits_list.append(
                seg_logits
            )  # (B, 1, num_classes, H, W) -> wait, head returns (B, NumClasses, H, W)

            # Pass tokens to next step
            # Head returns (B, N, C), exactly what we need for next input
            previous_temporal_tokens = next_temporal_tokens

            # NOTE: If running out of memory on long clips, uncomment below:
            # previous_temporal_tokens = previous_temporal_tokens.detach()

        # Stack outputs: (B, T, NumClasses, H, W)
        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def infer_image(self, image: torch.Tensor):
        """
        Inference with state maintenance using the internal deque buffer.
        """
        previous_temporal_tokens = None

        if (
            self.seg_head.use_temporal_consistency
            and len(self.temporal_token_buffer) > 0
        ):
            # Buffer stores tokens as (B, N, C). Stack them.
            # We treat the buffer as the history.
            # Flatten deque list -> (B, Total_N_Tokens, C)
            buffer_tokens = torch.cat(list(self.temporal_token_buffer), dim=1)
            previous_temporal_tokens = buffer_tokens

        seg_logits, temporal_tokens, _ = self.forward(image, previous_temporal_tokens)

        # Update buffer
        if self.seg_head.use_temporal_consistency and temporal_tokens is not None:
            # Output is (B, N, C). Add to buffer.
            self.temporal_token_buffer.append(temporal_tokens.detach())

        seg_probs = F.softmax(seg_logits, dim=1)
        segmentation_pred = torch.argmax(seg_probs, dim=1)

        return seg_probs, segmentation_pred.cpu().numpy()


def main():
    # TEST SCRIPT
    B = 2
    T = 3
    C = 3
    H = 476
    W = 630

    # Create fake clip
    clip = torch.randn(B, T, C, H, W).to("cpu")

    # Init Model
    model = Dino2Seg(
        use_temporal_consistency=True,
        num_temporal_tokens=4,
        temporal_window=3,
        device="cpu",
    )

    # Test Clip Forward (Training Mode)
    print("Testing forward_clip...")
    outputs = model.forward_clip(clip)
    print(f"Clip Output Shape: {outputs.shape}")  # Should be (B, T, Classes, H, W)

    # Test Inference Mode
    print("Testing inference...")
    model.reset_temporal_buffer()
    img = torch.randn(1, 3, H, W).to("cpu")
    probs, pred = model.infer_image(img)
    print(f"Inference Pred Shape: {pred.shape}")


if __name__ == "__main__":
    main()
