import argparse
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from enum import Enum

from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from ml4ded.util.vis import decode_segmap

class SegLabels(Enum):
    BACKGROUND = 0
    HEAD = 1
    BASEPLATE = 2
    PREVIOUS_PART = 3
    CURRENT_PART = 4
    WELD_FLASH = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Final Part Height Analysis per Frame with Overlay Scrubber')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory of input images')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights", help='Pretrained model weights directory')
    parser.add_argument('--expected-height-mm', type=float, required=True, help='Expected real-world height (mm) of the final part (CURRENT_PART)')
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

def main():
    args = parse_args()
    target_class = SegLabels.CURRENT_PART.value

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model
    sample_img = Image.open(next(iter(os.scandir(args.image_dir))).path).convert('RGB')
    img_w, img_h = make_divisible(sample_img.size)

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

    frame_heights = []
    frame_names = []
    overlays = []

    frames = sorted(os.scandir(args.image_dir), key=lambda x: x.name)

    for i, entry in enumerate(frames, 1):
        if not entry.name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        if i % 10 != 0:
            continue

        raw_img = Image.open(entry.path).convert('RGB')
        transformed_image = input_transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, seg_map = model.infer_image(transformed_image)

        seg_map_np = seg_map[0]
        height_px = compute_object_height(seg_map_np, target_class)

        if height_px is None:
            print(f"{entry.name}: No CURRENT_PART detected. Skipping.")
            continue

        frame_names.append(entry.name)
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

    frame_names = ["." + x.replace(x.split(".")[-1], "") for x in frame_names]
    frame_heights_mm = np.array(frame_heights) * scale_mm_per_px
    # Plot height vs frame
    axs[0].plot(range(len(frame_names)), frame_heights_mm, marker='o')
    axs[0].set_title('CURRENT_PART Height per Frame')
    axs[0].set_ylabel('Height (mm)')
    axs[0].set_xlabel('Image Name')
    axs[0].set_xticks(range(len(frame_names)))
    axs[0].set_xticklabels(frame_names, rotation=45, ha='right', fontsize=8)

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
    plt.show()

if __name__ == '__main__':
    main()
