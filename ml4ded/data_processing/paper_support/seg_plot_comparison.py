import os
import re
import argparse
import time

import numpy as np
import matplotlib

from ml4ded.util.dataset.get_model_weights import get_model_weights

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch

from ml4ded.models.dino2seg import Dino2Seg
from ml4ded.util.vis import decode_segmap

CLASS_NAMES = ["Background", "Head", "Baseplate", "Previous Part", "Current Part", "Weld Flash"]
CLASS_COLORS = [(0, 0, 0), (0, 128, 128), (0, 0, 128), (0, 128, 0), (128, 64, 128), (128, 0, 128)]
TEMPORAL_WINDOW = 4

def parse_filename(name):
    match = re.match(r"rgb_(\d+)_(\d+)\.png", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def load_label(path, size):
    return np.array(Image.open(path).resize(size, resample=Image.NEAREST))

def prepare_tensor(image_pil, img_w, img_h):
    transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return transform(image_pil).unsqueeze(0)  # [1, 3, H, W]

def load_temporal_stack(temporal_dir, video_id, frame_id, img_w, img_h, T):
    frames = []
    for i in range(frame_id - T, frame_id):
        fname = f"rgb_{video_id}_{i:05d}.png"
        path = os.path.join(temporal_dir, fname)
        if not os.path.exists(path):
            # Pad with first available
            path = os.path.join(temporal_dir, f"rgb_{video_id}_{max(i, 0):05d}.png")
        img = Image.open(path).convert("RGB")
        frames.append(prepare_tensor(img, img_w, img_h))  # [1, 3, H, W]

    return torch.stack(frames, dim=0)  # [T, 1, 3, H, W]

def run_model(model, image_tensor, device):
    with torch.no_grad():
        _, seg_map = model.infer_image(image_tensor)
    return seg_map[0]

def build_figure(data_dir, model_weights_dir, output_path=None):
    rgb_dir = os.path.join(data_dir, "rgb")
    seg_dir = os.path.join(data_dir, "seg")
    temporal_dir = os.path.join(data_dir, "temporal")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using device:", device)

    all_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")])
    all_files_info = [parse_filename(f) for f in all_files]
    all_files_info = [(v, f, i) for (v, i), f in zip(all_files_info, all_files) if v is not None]

    if not all_files_info:
        raise ValueError("No properly formatted RGB filenames found.")

    sample_img = Image.open(os.path.join(rgb_dir, all_files_info[0][1])).convert("RGB")
    img_w, img_h = make_divisible(sample_img.size)

    # Load models
    common_args = dict(
        encoder="vitb",
        num_classes=len(CLASS_NAMES),
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=model_weights_dir,
        use_clstoken=True,
        device=device,
    )

    baseline_model = Dino2Seg(**common_args, use_temporal_consistency=False, num_temporal_tokens=0, temporal_window=0, cross_attn_heads=0)
    temporal_model = Dino2Seg(**common_args, use_temporal_consistency=True, num_temporal_tokens=16, temporal_window=TEMPORAL_WINDOW, cross_attn_heads=4)

    baseline_model.eval().to(device)
    temporal_model.eval().to(device)

    row_labels = ["Image", "Label", "Baseline", "Temporal"]
    n_rows, n_cols = len(row_labels), len(all_files_info)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.8 * n_cols, 2.8 * n_rows),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for c, (vid_id, fname, frame_id) in enumerate(all_files_info):
        rgb_path = os.path.join(rgb_dir, fname)
        seg_path = os.path.join(seg_dir, fname.replace("rgb_", "seg_"))

        if not os.path.exists(seg_path):
            print(f"Skipping {fname}: segmentation not found")
            continue

        img = Image.open(rgb_path).convert("RGB")
        img_np = np.array(img.resize((img_w, img_h)))
        lbl_np = load_label(seg_path, (img_w, img_h))
        lbl_rgb = decode_segmap(lbl_np, nc=len(CLASS_NAMES))

        input_tensor = prepare_tensor(img, img_w, img_h).to(device)
        start_time = time.time()
        with torch.no_grad():
            pred_baseline = decode_segmap(run_model(baseline_model, input_tensor, device), nc=len(CLASS_NAMES))
        end_time = time.time()
        print(f"BASELINE - Vid: {vid_id} Frame: {frame_id} processed in: {(end_time - start_time):.3f} seconds")

        # Preload temporal frames (buffering)
        for i in range(frame_id - TEMPORAL_WINDOW, frame_id):
            fname = f"rgb_{vid_id}_{max(i, 0):05d}.png"
            temporal_path = os.path.join(temporal_dir, fname)
            if not os.path.exists(temporal_path):
                print(f"Missing temporal frame: {temporal_path}")
                continue
            temp_img = Image.open(temporal_path).convert("RGB")
            temp_tensor = prepare_tensor(temp_img, img_w, img_h).to(device)
            start_time = time.time()
            _ = temporal_model.infer_image(temp_tensor)  # Buffer only
            end_time = time.time()
            print(f"TEMPORAL BUFFERING - Vid: {vid_id} Frame: {frame_id} processed in: {(end_time - start_time):.3f} seconds")



        # Now run on the current frame
        with torch.no_grad():
            start_time = time.time()
            _, pred_temporal_map = temporal_model.infer_image(input_tensor)  # Uses internal buffer
            end_time = time.time()
            print(f"TEMPORAL - Vid: {vid_id} Frame: {frame_id} processed in: {(end_time - start_time):.3f} seconds")


        pred_temporal = decode_segmap(pred_temporal_map[0], nc=len(CLASS_NAMES))

        imgs = [img_np, lbl_rgb, pred_baseline, pred_temporal]

        for r, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=12, rotation=0, labelpad=40, ha='right', va='center')

        temporal_model.reset_temporal_buffer()

    # Adjust layout to leave room for the legend
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.20, wspace=0.02, hspace=0.02)

    # Create a dedicated axis below the images for the legend
    legend_ax = fig.add_axes([0.05, 0.02, 0.90, 0.12])  # [left, bottom, width, height]
    legend_ax.axis('off')

    fig.canvas.draw()
    square_width = 0.03
    square_height = 0.2
    text_offset = 0.005
    spacing = 0.02
    y_square = 0.5
    y_text = 0.6

    x_pos = 0.0
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        legend_ax.add_patch(plt.Rectangle(
            (x_pos, y_square), square_width, square_height,
            transform=legend_ax.transAxes,
            color=np.array(color) / 255.0, ec='none'
        ))

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
    parser.add_argument('--data-dir', type=str, default="./images/comparison_images", help='Root directory containing rgb/, seg/, and temporal/')
    parser.add_argument('--model-weights-dir', type=str, default="./model_weights", help='Pretrained model weights directory')
    parser.add_argument('--output-path', type=str, default="./images/comparison_images", help='Optional path to save figure')
    args = parser.parse_args()

    get_model_weights(args.model_weights_dir, use_temporal=True)
    get_model_weights(args.model_weights_dir, use_temporal=False)

    build_figure(
        args.data_dir,
        args.model_weights_dir,
        output_path=args.output_path
    )
