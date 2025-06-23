from depth_anything_v2.dinov2 import DinoVisionTransformer
from depth_anything_v2.dinov2_layers.dat import DAT


class DinoWithDAT(DinoVisionTransformer):
    def __init__(self, *args, dat_spatial_size=(37, 37), **kwargs):
        super().__init__(*args, **kwargs)

        self.dat = DAT(
            q_size=(37, 37),
            kv_size=(37, 37),
            n_heads=8,
            n_head_channels=96,
            n_groups=4,
            attn_drop=0.1,
            proj_drop=0.1,
            stride=1,
            offset_range_factor=1.0,
            use_pe=True,
            dwc_pe=False,
            no_off=False,
            fixed_pe=False,
            ksize=3,
            log_cpb=False,
        )

        self.dat_spatial_size = dat_spatial_size  # e.g., (518 / 14, 518 / 14)

    def forward_features(self, x, masks=None):
        features = super().forward_features(x, masks)

        if self.use_dat:
            patch_tokens = features["x_norm_patchtokens"]  # (B, N, D)
            B, N, D = patch_tokens.shape
            H, W = self.dat_spatial_size

            x_spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
            x_dat, _, _ = self.dat(x_spatial)

            features["x_dat_out"] = x_dat

        return features
