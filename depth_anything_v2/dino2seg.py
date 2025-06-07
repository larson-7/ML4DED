import math
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from depth_anything_v2.base import ConvBlock
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

# class Dino2Seg(nn.Module):
#     """
#     tokens ─► Linear ─► ReLU ─► Conv2d ─►  low‑res logits  (no upsample)
#     --------------------------------------------------------------------
#     * Use `upsample_logits()` (below) when you want H×W output.
#     """
#
#     def __init__(
#         self,
#         embed_dim: int = 768,       # ViT token dim (e.g. 768 for ViT‑B/14)
#         num_classes: int = 41,      # NYU = 41
#         hidden_dim: int = 512,      # width after Linear
#         patch_size: int = 14,       # backbone patch stride
#         image_height: int = 630,
#         image_width: int = 476,
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.mlp  = nn.Linear(embed_dim, hidden_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = ConvBlock(in_feature=hidden_dim, out_feature=num_classes)
#         self.image_height = image_height
#         self.image_width = image_width
#         self.ph = self.image_height // self.patch_size
#         self.pw = self.image_width // self.patch_size
#
#     # ----------------------------------------------------------------- #
#     @staticmethod
#     def _tokens_to_map(tokens, ph, pw, c):
#         """(B, N, C) → (B, C, ph, pw)"""
#         return tokens.permute(0, 2, 1).reshape(tokens.size(0), c, ph, pw)
#
#     # ----------------------------------------------------------------- #
#     def forward(self, tokens):
#         """
#         tokens: (B, N, C) patch embeddings from backbone
#         returns: (B, num_classes, ph, pw)  where ph = H/patch,  pw = W/patch
#         """
#         B, N, _ = tokens.shape
#
#         x = self.relu(self.mlp(tokens))        # (B, N, hidden)
#         x = self._tokens_to_map(x, self.ph, self.pw, x.size(-1))
#         logits_low = self.conv(x)              # (B, num_cls, ph, pw)
#         return logits_low
#
#
# # ──────────────────── one‑liner helper for later ───────────────────── #
# def upsample_logits(logits, target_hw, mode="bilinear"):
#     """
#     logits   : (B, C, h, w)   – low‑res output of SimpleSegHead
#     target_hw: (H, W)         – original image resolution
#     returns  : (B, C, H, W)   – ready for arg‑max or soft‑viz
#     """
#     return F.interpolate(logits, size=target_hw, mode=mode, align_corners=False)

class DPTSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels=768,                        # transformer embedding dim
        features=256,                           # decoder channels
        out_channels=[256, 512, 1024, 1024],    # for the 4 scale projections
        num_classes=40,                         # NYUDv2
        use_bn=False,
        use_clstoken=False
    ):
        super().__init__()
        self.use_clstoken = use_clstoken

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

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, num_classes, kernel_size=1)
        )

    def forward(self, out_features, ph, pw, upsample_hw=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]   # (B, N, C)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], ph, pw))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        seg = self.segmentation_head(path_1)
        # Upsample to target resolution if given (for full-res logits)
        if upsample_hw is not None:
            seg = F.interpolate(seg, upsample_hw, mode="bilinear", align_corners=True)
        return seg  # (B, num_classes, H, W)


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
            use_clstoken=False
,           model_weights_dir="",
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

        # load the weights into the model
        if model_weights_dir:
            model_weight_path = os.path.join(model_weights_dir, "dinov2_vitb14_reg4_pretrain.pth")
            print("loading pretrained weights from {}".format(model_weight_path))
            state_dict = torch.load(model_weight_path, map_location='cpu')
            missing_keys, unexpected_keys = self.pretrained.load_state_dict(state_dict, strict=False)
            # print warnings if keys don't match exactly
            if missing_keys:
                print("[Warning] Missing keys:", missing_keys)
            if unexpected_keys:
                print("[Warning] Unexpected keys:", unexpected_keys)

            self.pretrained.eval()
            # freeze weights for backbone
            for param in self.pretrained.parameters():
                param.requires_grad = False

        self.seg_head = DPTSegmentationHead(
            in_channels=features,
            num_classes=num_classes,
            out_channels=out_channels,
            use_bn = use_bn,
            use_clstoken = use_clstoken,
        ).to(self.device)

        self.pretrained.to(self.device)
        self.seg_head.to(self.device)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)

        depth = self.seg_head(out_features=features, ph=patch_h, pw=patch_w, upsample_hw = (self.image_height, self.image_width))
        depth = F.relu(depth)

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)

        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
