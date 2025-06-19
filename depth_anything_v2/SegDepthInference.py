import argparse
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from util.vis import decode_segmap


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--data-dir', type=str, default="../data/nyu_depth_v2",
                        help='train/test data directory')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights",
                        help='pretrained model weights directory')

    parser.add_argument('--base-size', type=int, default=580,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=518,
                        help='crop image size')

    # training hyper params
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')

    # checkpoint and log
    parser.add_argument('--save-dir', default='./ckpt',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--device', default='cuda',
                        help='Training device')
    args = parser.parse_args()
    return args


def make_divisible(val, divisor=14):
    return val - (val % divisor)

if __name__ == '__main__':
    args = parse_args()
    raw_img = Image.open("/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded/raw_images/buildplate000_1/003800.jpg")
    img_w, img_h = make_divisible(np.array(raw_img.size))

    model = SegmentationDeformableDepth(
        encoder="vitb",
        num_classes=6,
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=args.model_weights_dir,
    )
    model.eval()

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    transformed_image = input_transform(raw_img)
    transformed_image = transformed_image.unsqueeze(0).to(args.device)
    seg, depth = model(transformed_image)

    # Process segmentation output
    pred_labels = torch.max(seg, 1).indices
    seg_map = decode_segmap(pred_labels[0].detach().cpu().numpy())  # shape: (H, W, 3), uint8

    # Process depth output
    depth_map = depth[0].detach().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    # Invert: so closer = brighter
    depth_map_inv = 1.0 - depth_map

    # Apply colormap (PLASMA or INFERNO, etc.)
    depth_map_vis = cv2.applyColorMap((depth_map_inv * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Plot side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(raw_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(seg_map)
    axs[1].set_title('Predicted Segmentation')
    axs[1].axis('off')

    axs[2].imshow(depth_map_vis)
    axs[2].set_title('Predicted Depth')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


