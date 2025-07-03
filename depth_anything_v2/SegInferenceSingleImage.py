import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from enum import Enum

from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from util.vis import decode_segmap

class SegLabels(Enum):
    BACKGROUND = 0
    HEAD = 1
    BASEPLATE = 2
    PREVIOUS_PART = 3
    CURRENT_PART = 4
    WELD_FLASH = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Run segmentation and display overlay')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights", help='Pretrained model weights directory')
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def overlay_segmentation(image_np, seg_map, alpha=0.5):
    seg_color = decode_segmap(seg_map, nc=8)
    overlay = (alpha * seg_color + (1 - alpha) * image_np).astype(np.uint8)
    return overlay

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    raw_img = Image.open(args.image_path).convert('RGB')
    img_w, img_h = make_divisible(raw_img.size)

    # Load model
    model = SegmentationDeformableDepth(
        encoder="vitb",
        num_classes=6,
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=args.model_weights_dir,
        device=device,
    )
    model.eval().to(device)

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    transformed_image = input_transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, seg_map = model.infer_image(transformed_image)

    seg_map_np = seg_map[0]
    img_np = np.array(raw_img.resize((img_w, img_h)))
    overlay_img = overlay_segmentation(img_np, seg_map_np)

    # Display overlay
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay_img)
    plt.title("Segmentation Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
