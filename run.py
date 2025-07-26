import argparse
import os
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
import re

from ml4ded.util.dataset.get_model_weights import get_model_weights

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from enum import Enum
import matplotlib.cm as cm

from ml4ded.models.dino2seg import Dino2Seg
from ml4ded.util.vis import decode_segmap

class SegLabels(Enum):
    BACKGROUND = 0
    HEAD = 1
    BASEPLATE = 2
    PREVIOUS_PART = 3
    CURRENT_PART = 4
    WELD_FLASH = 5

def parse_args():
    parser = argparse.ArgumentParser(description="Run model on single image, video, or directory of images.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, default="", help='Path to a single input image')
    input_group.add_argument('--video', type=str, default="", help='Path to a video file')
    input_group.add_argument('--image-dir', type=str, default="", help='Directory of input images')
    parser.add_argument('--model-weights-dir', type=str, default="./model_weights", help='Pretrained model weights directory')
    parser.add_argument('--expected-height-mm', type=float, required=True, help='Expected real-world height (mm) of the final part (CURRENT_PART)')
    parser.add_argument('--enable-temporal', action='store_true', help='Enable temporal consistency')
    parser.add_argument('--frame-stride', default=15, help='number of frames to skip for each processed frame')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--color-layers', action='store_true', help='Color coordinate layers in plot')
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def compute_object_height(seg_map, target_class):
    mask = (seg_map == target_class)
    if not np.any(mask):
        return None
    ys = np.where(mask)[0]
    height_px = ys.max() - ys.min() + 1
    return height_px

def overlay_segmentation(image_np, seg_map, alpha=0.5):
    seg_color = decode_segmap(seg_map)
    overlay = (alpha * seg_color + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def group_layers_by_height(heights, threshold=0.2):
    """Group frame indices into layers based on height changes."""
    layers = []
    current_layer = [0]
    for i in range(1, len(heights)):
        if abs(heights[i] - heights[i-1]) > threshold:
            layers.append(current_layer)
            current_layer = [i]
        else:
            current_layer.append(i)
    layers.append(current_layer)
    return layers

def main():
    args = parse_args()
    target_class = SegLabels.CURRENT_PART.value
    frame_stride = 10
    device = args.device

    frames_to_process = []
    enable_temporal = args.enable_temporal

    if args.video:
        from ml4ded.util.img_vid_utils.video_cropping import crop_and_save_video
        video_path = args.video
        video_dir = os.path.dirname(video_path)
        tmp_img_dir = os.path.join(video_dir, 'tmp')
        os.makedirs(tmp_img_dir, exist_ok=True)
        _, frames_to_process = crop_and_save_video(input_video=video_path, output_dir=tmp_img_dir, stride=frame_stride)

    elif args.image_dir:
        dir_path = args.image_dir
        files = os.listdir(dir_path)

        # Extract numeric prefix (e.g., 60.1 from '60.1.png') and sort
        numbered_files = []
        for f in files:
            match = re.match(r"(\d+(?:\.\d+)?).*", f)
            if match:
                number = float(match.group(1))
                numbered_files.append((number, f))

        # Sort by the numeric value
        numbered_files.sort(key=lambda x: x[0])

        # Apply stride
        selected_files = [fname for _, fname in numbered_files[::frame_stride]]

        frames_to_process = [os.path.join(dir_path, f) for f in selected_files]
    elif args.image:
        frames_to_process = [args.image]
    else:
        print("No image, image directory, or video specified")
        exit(0)

    get_model_weights(args.model_weights_dir, use_temporal=enable_temporal)

    # Load model
    sample_img = Image.open(frames_to_process[0]).convert('RGB')
    img_w, img_h = make_divisible(sample_img.size)

    if enable_temporal:
        model = Dino2Seg(
            encoder="vitb",
            num_classes=len(SegLabels),
            image_height=img_h,
            image_width=img_w,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            model_weights_dir=args.model_weights_dir,
            use_clstoken=True,
            use_temporal_consistency=True,
            num_temporal_tokens=16,
            temporal_window=4,
            cross_attn_heads=4,
            device=device,
        )
    else:
        model = Dino2Seg(
            encoder="vitb",
            num_classes=len(SegLabels),
            image_height=img_h,
            image_width=img_w,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            model_weights_dir=args.model_weights_dir,
            use_clstoken=True,
            use_temporal_consistency=False,
            num_temporal_tokens=0,
            temporal_window=0,
            cross_attn_heads=0,
            device=device,
        )

    model.eval().to(device)

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    frame_heights = []
    frame_names = []
    overlays = []
    num_frames = len(frames_to_process)

    for i, entry in enumerate(frames_to_process, 1):
        if not entry.endswith(('.jpg', '.png', '.jpeg')):
            continue

        raw_img = Image.open(entry).convert('RGB')
        transformed_image = input_transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            _, seg_map = model.infer_image(transformed_image)
            end_time = time.time()
            print(f"Frame {i}/{num_frames} processed in: {(end_time - start_time):.3f} seconds")

        seg_map_np = seg_map[0]
        height_px = compute_object_height(seg_map_np, target_class)

        if height_px is None:
            print(f"{os.path.basename(entry)}: No CURRENT_PART detected. Skipping.")
            continue

        frame_names.append(entry)
        frame_heights.append(height_px)

        img_np = np.array(raw_img.resize((img_w, img_h)))
        overlay_img = overlay_segmentation(img_np, seg_map_np, alpha=0.5)
        overlays.append(overlay_img)

    print(f"\nTotal frames with CURRENT_PART detected: {len(frame_heights)}")
    if len(frame_heights) == 0:
        print("No frames with CURRENT_PART detected. Exiting.")
        return

    # Estimate final scale using last frame
    scale_mm_per_px = args.expected_height_mm / frame_heights[-1]
    print(f"Video: {os.path.basename(args.image_dir)}: Final Height_px = {frame_heights[-1]}, Scale ≈ {scale_mm_per_px:.3f} mm/px")

    # ---- Plotting with scrubber below ----
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
    plt.subplots_adjust(bottom=0.25)

    frame_names = [os.path.basename(x).replace("." + os.path.basename(x).split(".")[-1], "") for x in frame_names]
    frame_heights_mm = np.array(frame_heights) * scale_mm_per_px

    if args.color_layers:
    # Group frames into layers and assign colors
        threshold = 0.2  # mm, adjust as needed for your process
        layers = group_layers_by_height(frame_heights_mm, threshold=threshold)
        num_layers = len(layers)
        colors = cm.get_cmap('tab20', num_layers)

        # Plot height vs frame, color by layer
        for idx, layer in enumerate(layers):
            axs[0].plot(
                layer,
                frame_heights_mm[layer],
                marker='o',
                color=colors(idx),
                label=f'Layer {idx+1}'
            )
            # Optionally, plot mean line for each layer
            mean_height = np.mean(frame_heights_mm[layer])
            axs[0].plot(
                [layer[0], layer[-1]],
                [mean_height, mean_height],
                color=colors(idx),
                linestyle='--',
                linewidth=2
            )
    else:
        axs[0].plot(range(len(frame_names)), frame_heights_mm, marker='o')  

    axs[0].set_title('CURRENT_PART Height per Frame')
    axs[0].set_ylabel('Height (mm)')
    axs[0].set_xlabel('Image Name')
    axs[0].set_xticks(range(len(frame_names)))
    axs[0].set_xticklabels(frame_names, rotation=45, ha='right', fontsize=8)
    axs[0].legend(loc='best', fontsize=8)

    # Display initial overlay
    overlay_ax = axs[1]
    overlay_img_obj = overlay_ax.imshow(overlays[0])
    overlay_ax.axis('off')
    overlay_ax.set_title(f"Frame: {frame_names[0]}")

    # Slider below both plots
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame Index',
        valmin=0,
        valmax=len(overlays) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        idx = int(frame_slider.val)
        overlay_img_obj.set_data(overlays[idx])
        overlay_ax.set_title(f"Frame: {frame_names[idx]}")
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    plt.tight_layout()
    plt.savefig("no_temp_output_plot.png")
    plt.show()

    # ─────────────── CLEANUP TEMPORARY DIRECTORY ─────────────── #
    if args.video:
        try:
            shutil.rmtree(tmp_img_dir)
            print(f"Temporary directory {tmp_img_dir} deleted.")
        except Exception as e:
            print(f"Failed to delete temporary directory {tmp_img_dir}: {e}")

if __name__ == '__main__':
    main()