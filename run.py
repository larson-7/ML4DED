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
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from enum import Enum
import matplotlib.cm as cm
from scipy.ndimage import median_filter, label

from ml4ded.models.dino2seg import Dino2Seg
from ml4ded.util.vis import decode_segmap
from filterpy.kalman import KalmanFilter


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
    parser.add_argument('--frame-stride', default=15, type=int, help='number of frames to skip for each processed frame')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--color-layers', action='store_true', help='Color coordinate layers in plot')
    parser.add_argument('--use-median', action='store_true', help='Apply median filtering to height measurements')
    parser.add_argument('--filter-window', default=7, type=int, help='Window size for median filter (odd number)')
    parser.add_argument('--confidence-threshold', type=float, default=0.8, 
                        help='Minimum confidence score for segmentation (0.0-1.0)')
    # Add this argument in parse_args()
    parser.add_argument('--min-area-pct', type=float, default=0.0,
                   help='Minimum percentage of total area for a component to be considered valid (0.0-1.0)')
    parser.add_argument('--show-main-component', action='store_true',
                   help='Only show the main connected component in visualizations')
    
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

# Add this import at the top


# Replace compute_object_height with this improved version:
def compute_object_height(seg_map, target_class, conf_map=None, threshold=0.5, min_area_pct=0.05):
    """
    Compute object height considering only the largest connected component.
    
    Args:
        seg_map: Segmentation map with class indices
        target_class: Class index to measure
        conf_map: Optional confidence map with scores 0-1
        threshold: Minimum confidence score to consider valid
        min_area_pct: Minimum percentage of total area to consider valid (filters noise)
    """
    mask = (seg_map == target_class)
    if not np.any(mask):
        return None
        
    # Apply confidence threshold if provided
    if conf_map is not None:
        # Only keep pixels where confidence exceeds threshold
        valid_mask = mask & (conf_map > threshold)
        if not np.any(valid_mask):
            print(f"Warning: No pixels with confidence > {threshold} for class {target_class}")
            return None
        mask = valid_mask
    
    # Find connected components
    labeled_mask, num_components = label(mask)
    if num_components == 0:
        return None
        
    # Get the size of each component
    component_sizes = np.bincount(labeled_mask.ravel())[1:] if num_components > 0 else []
    if len(component_sizes) == 0:
        return None
        
    # Find the largest component
    largest_component_label = np.argmax(component_sizes) + 1
    largest_component_size = component_sizes[largest_component_label-1]
    
    # Filter out small components (likely noise)
    total_area = np.sum(mask)
    if largest_component_size < (min_area_pct * total_area) and num_components > 1:
        print(f"Warning: Largest component too small ({largest_component_size}/{total_area} pixels)")
        return None
        
    # Create a mask with only the largest component
    main_component_mask = (labeled_mask == largest_component_label)
    
    # Print component information
    print(f"Found {num_components} components for class {target_class}")
    print(f"Using largest component: {largest_component_size} pixels ({largest_component_size/total_area:.1%} of total)")
    
    # Calculate height from largest component only
    ys = np.where(main_component_mask)[0]
    height_px = ys.max() - ys.min() + 1
    return height_px

def overlay_segmentation(image_np, seg_map, alpha=0.5):
    seg_color = decode_segmap(seg_map)
    overlay = (alpha * seg_color + (1 - alpha) * image_np).astype(np.uint8)
    return overlay

def download_weights_if_needed(model_weights_dir, use_temporal, repo_id="iknocodes/ml4ded"):
    from huggingface_hub import hf_hub_download
    vitb_model_name = "dinov2_vitb14_reg4_pretrain.pth"
    os.makedirs(model_weights_dir, exist_ok=True)
    # Check for backbone
    vitb_weight_file = os.path.join(model_weights_dir, vitb_model_name)
    if not os.path.exists(vitb_weight_file):
        print("Downloading ViT-b backbone weights from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename=vitb_model_name, local_dir=model_weights_dir, local_dir_use_symlinks=False)
    # Seg head
    if use_temporal:
        seg_file = "ml4ded_seg_temporal.pth"
    else:
        seg_file = "ml4ded_seg.pth"
    seg_weight_file = os.path.join(model_weights_dir, seg_file)
    if not os.path.exists(seg_weight_file):
        print(f"Downloading {seg_file} from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename=seg_file, local_dir=model_weights_dir, local_dir_use_symlinks=False)

def group_layers_by_height(heights, threshold=0.3):
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
    frame_stride = args.frame_stride
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

    download_weights_if_needed(args.model_weights_dir, use_temporal=enable_temporal)

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
            num_temporal_tokens=24,
            temporal_window=16,
            cross_attn_heads=16,
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
            # Get both segmentation probabilities and predictions
            seg_probs, seg_map = model.infer_image(transformed_image)
            end_time = time.time()
            print(f"Frame {i}/{num_frames} processed in: {(end_time - start_time):.3f} seconds")

        seg_map_np = seg_map[0]
        # Get confidence scores for the current_part class
        confidence_map = seg_probs[0, target_class].cpu().numpy()
        target_mask = (seg_map_np[0] == target_class)
        if np.any(target_mask):
            avg_confidence = np.mean(confidence_map[target_mask])
            max_confidence = np.max(confidence_map[target_mask])
            num_pixels = np.sum(target_mask)
            print(f"CURRENT_PART confidence - avg: {avg_confidence:.3f}, max: {max_confidence:.3f}, pixels: {num_pixels}")
        else:
            print(f"No CURRENT_PART pixels detected in frame")

        if enable_temporal:
            height_px = compute_object_height(seg_map_np, target_class, 
                                            conf_map=confidence_map, 
                                            threshold=args.confidence_threshold,
                                            min_area_pct=args.min_area_pct)
        else:
            height_px = compute_object_height(seg_map_np, target_class, 
                                            conf_map=confidence_map, 
                                            threshold=args.confidence_threshold,
                                            min_area_pct=0.0) 

        if height_px is None:
            print(f"{os.path.basename(entry)}: No CURRENT_PART detected. Skipping.")
            continue

        frame_names.append(entry)
        frame_heights.append(height_px)

        img_np = np.array(raw_img.resize((img_w, img_h)))
        if args.show_main_component:
        # Create a filtered segmentation map with only the main component
            filtered_seg_map = seg_map_np.copy()
            
            # Find the target class mask and its components again
            mask = (seg_map_np == target_class)
            labeled_mask, num_components = label(mask)
            
            if num_components > 1:
                # Get component sizes
                component_sizes = np.bincount(labeled_mask.ravel())[1:]
                largest_component_label = np.argmax(component_sizes) + 1
                
                # Create a mask for all areas to clear
                areas_to_clear = mask & (labeled_mask != largest_component_label)
                
                # Set these areas to background (0)
                filtered_seg_map[areas_to_clear] = 0
                
                print(f"Visualization: Removed {num_components-1} smaller components")
            
            # Use filtered map for overlay
            overlay_img = overlay_segmentation(img_np, filtered_seg_map, alpha=0.5)
        else:
            # Standard segmentation overlay with all components
            overlay_img = overlay_segmentation(img_np, seg_map_np, alpha=0.5)
            
        overlays.append(overlay_img)

    print(f"\nTotal frames with CURRENT_PART detected: {len(frame_heights)}")
    if len(frame_heights) == 0:
        print("No frames with CURRENT_PART detected. Exiting.")
        return
    
    # Store original heights before any filtering
    original_heights = np.array(frame_heights).copy()

    if len(frame_heights) > 1:
    # Apply median filter if requested
        if args.use_median:
            window_size = args.filter_window
            if window_size % 2 == 0:  # Ensure window size is odd
                window_size += 1
            print(f"Applying median filter with window size {window_size}...")
            frame_heights = median_filter(frame_heights, size=window_size)
        
        
        # Print stats about filtering if any filtering was applied
        if args.use_median:
            changes = np.abs(original_heights - frame_heights)
            if np.any(changes > 0):
                avg_change = np.mean(changes[changes > 0])
                max_change = np.max(changes)
                num_changed = np.sum(changes > 0)
                print(f"Filtering changed {num_changed}/{len(frame_heights)} values")
                print(f"Average change: {avg_change:.2f}px, Max change: {max_change:.2f}px")
    # Estimate final scale using last frame
    scale_mm_per_px = args.expected_height_mm / frame_heights[-1]
    print(f"Video: {os.path.basename(args.image_dir if args.image_dir else args.video if args.video else args.image)}: Final Height_px = {frame_heights[-1]}, Scale ≈ {scale_mm_per_px:.3f} mm/px")

    # ---- Plotting with scrubber below ----
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
    plt.subplots_adjust(bottom=0.25)

    frame_names = [os.path.basename(x).replace("." + os.path.basename(x).split(".")[-1], "") for x in frame_names]
    frame_heights_mm = np.array(frame_heights) * scale_mm_per_px
    original_heights_mm = original_heights * scale_mm_per_px

    if args.color_layers:
    # First, plot a continuous line connecting all points
        heights_for_layers = frame_heights_mm
        axs[0].plot(
            range(len(frame_heights_mm)),
            frame_heights_mm,
            linestyle='-',
            color='lightgray',
            alpha=0.5,
            zorder=1  # Make sure this is drawn below the colored markers
        )
        
        # Define layers based on height increases
        layers = []
        current_layer = [0]

        threshold = 0.2  # mm
        layers = []
        current_layer = [0]

        for i in range(1, len(heights_for_layers)):
            current_avg = np.mean([heights_for_layers[j] for j in current_layer])
            if heights_for_layers[i] > current_avg + threshold:
                layers.append(current_layer)
                current_layer = [i]
            else:
                current_layer.append(i)
        
        # Add the last layer
        if current_layer:
            layers.append(current_layer)
            
        num_layers = len(layers)
        colors = cm.get_cmap('tab20', num_layers)
        
        for idx, layer_indices in enumerate(layers):
            axs[0].scatter(
                layer_indices,
                heights_for_layers[layer_indices],
                marker='o',
                color=colors(idx),
                label=f'Layer {idx+1}',
                s=50,  
                zorder=2  
            )
    else:
        if args.use_median:
            changes = np.abs(original_heights - frame_heights)
            if np.any(changes > 0):
                avg_change = np.mean(changes[changes > 0])
                max_change = np.max(changes)
                num_changed = np.sum(changes > 0)
                print(f"Filtering changed {num_changed}/{len(frame_heights)} values")
                print(f"Average change: {avg_change:.2f}px, Max change: {max_change:.2f}px")
            
            if enable_temporal:
                filter_label = f"Our Model"
                axs[0].plot(range(len(frame_names)), frame_heights_mm, 
                            marker='o', linestyle='-', 
                            color='blue', label=filter_label)
            else:
                axs[0].plot(range(len(frame_names)), original_heights_mm, 
                            marker='.', linestyle='-', alpha=0.6, 
                            color='gray', label='Baseline model')
                
        else:
            if enable_temporal:
                axs[0].plot(range(len(frame_names)), frame_heights_mm, 
                            marker='o', linestyle='-', 
                            color='blue', label='Temporal model')
            else:
                axs[0].plot(range(len(frame_names)), frame_heights_mm, 
                            marker='o', linestyle='-', 
                            color='blue', label='Baseline model')
    title_suffix = ""
    if enable_temporal:
        title_suffix += " (Temporal)"

    axs[0].set_title('CURRENT_PART Height per Frame')
    axs[0].set_ylabel('Height (mm)')
    axs[0].set_xlabel('Frame')
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

    if args.image_dir:
    # Extract directory name from path (e.g., get '5' from 'layer_images/5/')
        dir_name = os.path.basename(os.path.normpath(args.image_dir))
        file_prefix = f"{dir_name}_"
    elif args.video:
        # Use video filename without extension
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        file_prefix = f"{video_name}_"
    elif args.image:
        # Use image filename without extension
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        file_prefix = f"{img_name}_"
    else:
        file_prefix = ""

    plt.tight_layout()
    filters_used = []
    if args.use_median:
        filters_used.append("median")
    if enable_temporal:
        filters_used.append("temporal")
        
    filename_suffix = "_".join(filters_used) if filters_used else "unfiltered"
    plt.savefig(f"{file_prefix}height_plot_{filename_suffix}.png")
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