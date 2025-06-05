import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.base import ConvBlock


class Dino2Seg(nn.Module):
    """
    tokens ─► Linear ─► ReLU ─► Conv2d ─►  low‑res logits  (no upsample)
    --------------------------------------------------------------------
    * Use `upsample_logits()` (below) when you want H×W output.
    """

    def __init__(
        self,
        embed_dim: int = 768,       # ViT token dim (e.g. 768 for ViT‑B/14)
        num_classes: int = 41,      # NYU = 41
        hidden_dim: int = 512,      # width after Linear
        patch_size: int = 14,       # backbone patch stride
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mlp  = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv = ConvBlock(in_feature=hidden_dim, out_feature=num_classes)

    # ----------------------------------------------------------------- #
    @staticmethod
    def _tokens_to_map(tokens, ph, pw, c):
        """(B, N, C) → (B, C, ph, pw)"""
        return tokens.permute(0, 2, 1).reshape(tokens.size(0), c, ph, pw)

    # ----------------------------------------------------------------- #
    def forward(self, tokens):
        """
        tokens: (B, N, C) patch embeddings from backbone
        returns: (B, num_classes, ph, pw)  where ph = H/patch,  pw = W/patch
        """
        B, N, _ = tokens.shape
        ph = pw = int(math.sqrt(N))            # works for square grids

        x = self.relu(self.mlp(tokens))        # (B, N, hidden)
        x = self._tokens_to_map(x, ph, pw, x.size(-1))
        logits_low = self.conv(x)              # (B, num_cls, ph, pw)
        return logits_low


# ──────────────────── one‑liner helper for later ───────────────────── #
def upsample_logits(logits, target_hw, mode="bilinear"):
    """
    logits   : (B, C, h, w)   – low‑res output of SimpleSegHead
    target_hw: (H, W)         – original image resolution
    returns  : (B, C, H, W)   – ready for arg‑max or soft‑viz
    """
    return F.interpolate(logits, size=target_hw, mode=mode, align_corners=False)

