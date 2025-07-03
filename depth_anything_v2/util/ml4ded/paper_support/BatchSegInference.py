import argparse
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pandas as pd
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
    parser = argparse.ArgumentParser(description='Batch Part Height Analysis Across Videos with Excel Config')
    parser.add_argument('--root-dir', type=str, required=True, help='Root directory containing video folders')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights", help='Pretrained model weights directory')
    parser.add_argument('--xlsx-path', type=str, required=True, help='Excel file (.xlsx) with sample metadata')
    parser.add_argument('--frame-skip', type=int, default=10, help='Process every Nth frame')
    parser.add_argument('--mode', type=str, choices=['final_height', 'insitu'], default='final_height',
                        help='Whether to compute final frame scale only or full in-situ analysis')
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

def get_sample_metadata(sample_name, xlsx_df):
    SAMPLE_COL = 'Sample Number'
    HEIGHT_COL = 'Average Actual Height [mm]'
    LAYER_COL = 'Layer Height [mm]'

    for col in [SAMPLE_COL, HEIGHT_COL, LAYER_COL]:
        if col not in xlsx_df.columns:
            raise ValueError(f"Excel file is missing required column: '{col}'")

    parsed_sample_numbers = xlsx_df[SAMPLE_COL].str.split('_').str[-1]
    row = xlsx_df[parsed_sample_numbers == sample_name]
    if row.empty:
        return None

    expected_height = float(row[HEIGHT_COL].values[0])
    layer_height = float(row[LAYER_COL].values[0])
    return expected_height, layer_height

def process_video_folder(video_dir, sample_name, expected_height_mm, layer_height_mm,
                         model, input_transform, device, frame_skip, mode):
    target_class = SegLabels.CURRENT_PART.value
    frame_heights = []
    frame_names = []
    overlays = []

    frames = sorted([
        entry for entry in os.scandir(video_dir)
        if entry.name.lower().endswith(('.jpg', '.png', '.jpeg'))
    ], key=lambda x: x.name)

    if mode == 'final_height':
        if len(frames) == 0:
            print(f"{sample_name}: No frames found.")
            return None

        # Search backwards from the last frame until part is detected
        found_frame = None
        for entry in reversed(frames):
            raw_img = Image.open(entry.path).convert('RGB')
            transformed_image = input_transform(raw_img).unsqueeze(0).to(device)

            with torch.no_grad():
                _, seg_map = model.infer_image(transformed_image)

            seg_map_np = seg_map[0]
            height_px = compute_object_height(seg_map_np, target_class)

            if height_px is not None:
                found_frame = (entry.name, height_px)
                break

        if found_frame is None:
            print(f"{sample_name}: No CURRENT_PART detected in any frame.")
            return None

        frame_name, height_px = found_frame
        scale_mm_per_px = expected_height_mm / height_px
        print(f"{sample_name}: Found in frame '{frame_name}' - Expected Height(mm)={expected_height_mm}, Height_px={height_px}, Scale ≈ {scale_mm_per_px:.3f} mm/px")
        return sample_name, scale_mm_per_px, layer_height_mm

    else:  # insitu mode
        frames_to_process = [f for i, f in enumerate(frames, 1) if i % frame_skip == 0]

        for entry in frames_to_process:
            raw_img = Image.open(entry.path).convert('RGB')
            transformed_image = input_transform(raw_img).unsqueeze(0).to(device)

            with torch.no_grad():
                _, seg_map = model.infer_image(transformed_image)

            seg_map_np = seg_map[0]
            height_px = compute_object_height(seg_map_np, target_class)

            if height_px is not None:
                frame_names.append(entry.name)
                frame_heights.append(height_px)

                img_np = np.array(raw_img.resize((model.image_width, model.image_height)))
                overlay_img = overlay_segmentation(img_np, seg_map_np, alpha=0.5)
                overlays.append(overlay_img)

        if len(frame_heights) == 0:
            print(f"{sample_name}: No CURRENT_PART detected.")
            return None

        scale_mm_per_px = expected_height_mm / frame_heights[-1]
        frame_heights_mm = np.array(frame_heights) * scale_mm_per_px

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
        plt.subplots_adjust(bottom=0.25)

        frame_names_display = ["." + x.replace(x.split(".")[-1], "") for x in frame_names]
        axs[0].plot(range(len(frame_names_display)), frame_heights_mm, marker='o')
        axs[0].set_title(f'{sample_name} - CURRENT_PART Height per Frame')
        axs[0].set_ylabel('Height (mm)')
        axs[0].set_xlabel('Frame Index')
        axs[0].set_xticks(range(len(frame_names_display)))
        axs[0].set_xticklabels(frame_names_display, rotation=45, ha='right', fontsize=8)

        overlay_ax = axs[1]
        overlay_img_obj = overlay_ax.imshow(overlays[0])
        overlay_ax.axis('off')
        overlay_ax.set_title(f"Frame: {frame_names_display[0]}")

        from matplotlib.widgets import Slider
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
            overlay_ax.set_title(f"Frame: {frame_names_display[idx]}")
            fig.canvas.draw_idle()

        frame_slider.on_changed(update)

        plt.tight_layout()
        plt.show()

        return None

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    xlsx_df = pd.read_excel(args.xlsx_path)

    first_video_dir = next(
        (os.path.join(args.root_dir, d.name) for d in os.scandir(args.root_dir) if d.is_dir()),
        None
    )
    if first_video_dir is None:
        print("No video folders found.")
        return

    sample_img_path = next(
        (entry.path for entry in os.scandir(first_video_dir) if entry.name.lower().endswith(('.jpg', '.png', '.jpeg'))),
        None
    )
    if sample_img_path is None:
        print(f"No images found in {first_video_dir}.")
        return

    sample_img = Image.open(sample_img_path).convert('RGB')
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

    final_height_results = []

    for video_dir_entry in os.scandir(args.root_dir):
        if not video_dir_entry.is_dir():
            continue

        sample_name = video_dir_entry.name
        sample_metadata = get_sample_metadata(sample_name, xlsx_df)
        if sample_metadata is None:
            print(f"Skipping {sample_name}: Not found in Excel sheet.")
            continue

        expected_height_mm, layer_height_mm = sample_metadata

        result = process_video_folder(
            video_dir=video_dir_entry.path,
            sample_name=sample_name,
            expected_height_mm=expected_height_mm,
            layer_height_mm=layer_height_mm,
            model=model,
            input_transform=input_transform,
            device=device,
            frame_skip=args.frame_skip,
            mode=args.mode,
        )

        if result is not None:
            final_height_results.append(result)

    if args.mode == 'final_height' and final_height_results:
        sample_names, avg_scales, layer_heights = zip(*final_height_results)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(sample_names, avg_scales, color='skyblue')
        ax.set_ylabel('Scale (mm/px)')
        ax.set_xlabel('Sample')
        ax.set_title('Final Height Scale per Sample')
        ax.set_xticklabels(sample_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
