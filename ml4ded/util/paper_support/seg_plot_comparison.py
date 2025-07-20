import matplotlib
matplotlib.use('tkAgg')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch

from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from depth_anything_v2.util.vis import decode_segmap

CLASS_NAMES = [
    "Background", "Head", "Baseplate", "Previous Part", "Current Part", "Weld Flash",
]
CLASS_COLORS = [
    (0, 0, 0), (0, 128, 128), (0, 0, 128), (0, 128, 0),
    (128, 64, 128), (128, 0, 128)
]

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def load_label(label_path, size):
    label = Image.open(label_path).resize(size, resample=Image.NEAREST)
    return np.array(label)

def run_model(model, image_tensor, device):
    with torch.no_grad():
        _, seg_map = model.infer_image(image_tensor)
    return seg_map[0]

def prepare_tensor(image_pil, img_w, img_h):
    transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return transform(image_pil).unsqueeze(0)

def build_figure(image_dir, model_weights_dir, output_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Collect image files
    all_files = os.listdir(image_dir)
    rgb_images = sorted([f for f in all_files if f.startswith("rgb_") and f.endswith(".png")])
    if not rgb_images:
        raise ValueError("No rgb_*.png images found in directory.")

    # Load one image to determine size
    sample = Image.open(os.path.join(image_dir, rgb_images[0])).convert("RGB")
    img_w, img_h = make_divisible(sample.size)

    # Load model
    model = SegmentationDeformableDepth(
        encoder="vitb", num_classes=6,
        image_height=img_h, image_width=img_w,
        features=768, out_channels=[256, 512, 1024, 1024],
        model_weights_dir=model_weights_dir,
        device=device
    )
    model.eval().to(device)

    row_labels = ["Image", "Label", "Baseline"]
    n_rows, n_cols = len(row_labels), len(rgb_images)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.0 * n_cols, 5.5),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.subplots_adjust(left=0.2, bottom=0.25)

    # Ensure axes is 2D array
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through and fill plots
    for c, fn in enumerate(rgb_images):
        rgb_p = os.path.join(image_dir, fn)
        seg_p = os.path.join(image_dir, fn.replace("rgb_", "seg_"))
        if not os.path.exists(seg_p):
            print(f"skip {fn}, no label")
            continue

        img = Image.open(rgb_p).convert("RGB")
        img_np = np.array(img.resize((img_w, img_h)))
        lbl_np = load_label(seg_p, (img_w, img_h))
        lbl_rgb = decode_segmap(lbl_np, nc=6)

        t = prepare_tensor(img, img_w, img_h).to(device)
        pred = run_model(model, t, device)
        pred_rgb = decode_segmap(pred, nc=6)

        for r, im in enumerate([img_np, lbl_rgb, pred_rgb]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=12,
                              rotation=0, labelpad=40,
                              ha='right', va='center')

    # Build horizontal legend
    legend_ax = fig.add_axes([0.10, 0.02, 0.80, 0.12])
    legend_ax.axis('off')

    fig.canvas.draw()  # Required for measuring text extents

    square_width = 0.03
    square_height = 0.2
    text_offset = 0.005
    spacing = 0.02
    y_square = 0.5
    y_text = 0.6

    x_pos = 0.0
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        # Draw colored square
        legend_ax.add_patch(plt.Rectangle(
            (x_pos, y_square), square_width, square_height,
            transform=legend_ax.transAxes,
            color=np.array(color) / 255.0, ec='none'
        ))

        # Draw label
        text_obj = legend_ax.text(x_pos + square_width + text_offset, y_text, name,
                                  transform=legend_ax.transAxes,
                                  fontsize=9, va='center', ha='left')

        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = legend_ax.transAxes.inverted()
        bbox_ax = inv.transform(bbox)
        text_width_ax = bbox_ax[1][0] - bbox_ax[0][0]

        x_pos += square_width + text_offset + text_width_ax + spacing

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing rgb_*.png and seg_*.png')
    parser.add_argument('--model-weights-dir', type=str, required=True, help='Pretrained model weights dir')
    parser.add_argument('--output-path', type=str, default=None, help='Optional path to save figure')
    args = parser.parse_args()

    build_figure(
        args.image_dir,
        args.model_weights_dir,
        output_path=args.output_path
    )
