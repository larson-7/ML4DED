import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .util.blocks import FeatureFusionBlock, _make_scratch


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

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        use_temporal_consistency=False,
        num_temporal_tokens=1,
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken

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

        # temporal consistency tokens
        self.use_temporal_consistency = use_temporal_consistency
        if self.use_temporal_consistency:
            self.num_temporal_tokens = num_temporal_tokens

            # how many patches each temporal token gets
            num_temporal_patch_channels = in_channels // num_temporal_tokens

            self.temporal_token_extraction = nn.ModuleList()
            for i in range(self.num_temporal_tokens):
                self.temporal_token_extraction.append(
                    nn.Sequential(
                        nn.Conv2d(num_temporal_patch_channels, 1, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(True),
                    )
                )


    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                # create readout which is a copy of the cls token for each spatial token
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        if self.use_temporal_consistency:
            B, C, H, W = layer_4.shape
            assert H >= self.num_temporal_tokens, "Cannot have more temporal tokens than patch rows"

            # Divide into `n` vertical chunks (along height axis)
            chunks = torch.chunk(layer_4, self.num_temporal_tokens, dim=2)  # list of (B, C, H_chunk, W)

            temporal_tokens = []
            for i, chunk in enumerate(chunks):
                # Pool over spatial dimensions to get one token per chunk
                temporal_out = self.temporal_token_extraction[i](chunk)
                token = temporal_out.view(B, C)  # (B, C)
                temporal_tokens.append(token)

            # Stack: (B, n, C)
            temporal_tokens = torch.stack(temporal_tokens, dim=1)
        # todo: no need to cast temporal tokens onto spatial tokens, the cross-attention block next will do that.
        #Need to add the N frames of temporal tokens as an input to this forward call

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

        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)
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
