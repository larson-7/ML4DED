import argparse
import os
import re
import json
import shutil
import numpy as np

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from datetime import datetime
from enum import Enum

from ml4ded.models.dino2seg import Dino2Seg
from utils.image_processing import process_frames, apply_filtering
from utils.layer_detection import group_layers_by_height, convert_layers_to_ids
from utils.visualization import create_visualization, get_output_filename
from utils.metrics import compute_temporal_metrics, analyze_layer_metrics, detect_anomalies

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
    parser.add_argument('--frame-stride', default=10, type=int, help='number of frames to skip for each processed frame')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--color-layers', action='store_true', help='Color coordinate layers in plot')
    parser.add_argument('--use-median', action='store_true', help='Apply median filtering to height measurements')
    parser.add_argument('--filter-window', default=7, type=int, help='Window size for median filter (odd number)')
    parser.add_argument('--confidence-threshold', type=float, default=0.8, 
                        help='Minimum confidence score for segmentation (0.0-1.0)')
    parser.add_argument('--min-area-pct', type=float, default=0.0,
                   help='Minimum percentage of total area for a component to be considered valid (0.0-1.0)')
    parser.add_argument('--show-main-component', action='store_true',
                   help='Only show the main connected component in visualizations')
    parser.add_argument('--save-metrics', type=str, help='Path to save metrics JSON file')
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def download_weights_if_needed(model_weights_dir, use_temporal, repo_id="iknocodes/ml4ded"):
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

def load_input_frames(args):
    """Load input frames from image, image directory, or video."""
    frames_to_process = []
    frame_stride = args.frame_stride
    
    if args.video:
        from ml4ded.util.img_vid_utils.video_cropping import crop_and_save_video
        video_path = args.video
        video_dir = os.path.dirname(video_path)
        tmp_img_dir = os.path.join(video_dir, 'tmp')
        os.makedirs(tmp_img_dir, exist_ok=True)
        _, frames_to_process = crop_and_save_video(input_video=video_path, output_dir=tmp_img_dir, stride=frame_stride)
        return frames_to_process, tmp_img_dir
        
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
        
    return frames_to_process, None

def initialize_model(image_path, model_weights_dir, enable_temporal, device):
    """Initialize the segmentation model."""
    sample_img = Image.open(image_path).convert('RGB')
    img_w, img_h = make_divisible(sample_img.size)

    if enable_temporal:
        model = Dino2Seg(
            encoder="vitb",
            num_classes=len(SegLabels),
            image_height=img_h,
            image_width=img_w,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            model_weights_dir=model_weights_dir,
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
            model_weights_dir=model_weights_dir,
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
    
    return model, input_transform, (img_w, img_h)

def save_metrics(metrics, args):
    """Save metrics to a JSON file if requested."""
    if hasattr(args, 'save_metrics') and args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"\nMetrics saved to: {args.save_metrics}")

def main():
    args = parse_args()
    target_class = SegLabels.CURRENT_PART.value
    
    # Load input frames
    frames_to_process, tmp_dir = load_input_frames(args)
    if not frames_to_process:
        print("No valid frames to process. Exiting.")
        return
    
    # Prepare model
    download_weights_if_needed(args.model_weights_dir, use_temporal=args.enable_temporal)
    model, input_transform, (img_w, img_h) = initialize_model(
        frames_to_process[0], 
        args.model_weights_dir, 
        args.enable_temporal, 
        args.device
    )
    
    # Process frames
    frame_heights, frame_names, overlays, confidence_scores = process_frames(
        model, 
        input_transform, 
        frames_to_process, 
        args, 
        target_class,
        (img_w, img_h)  
    )
    
    print(f"\nTotal frames with CURRENT_PART detected: {len(frame_heights)}")
    if len(frame_heights) == 0:
        print("No frames with CURRENT_PART detected. Exiting.")
        if tmp_dir and args.video:
            shutil.rmtree(tmp_dir)
        return
    
    # Apply filtering if requested
    frame_heights, original_heights = apply_filtering(frame_heights, args)
    
    # Calculate scale
    scale_mm_per_px = args.expected_height_mm / frame_heights[-1]
    frame_heights_mm = frame_heights * scale_mm_per_px
    original_heights_mm = original_heights * scale_mm_per_px
    print(f"Scale: {scale_mm_per_px:.4f} mm/px")
    
    # Detect layers
    layers = group_layers_by_height(frame_heights_mm, threshold=0.2)
    num_layers = len(layers)
    layer_ids = convert_layers_to_ids(layers, len(frame_heights_mm))
    
    # Calculate metrics
    temporal_metrics = compute_temporal_metrics(frame_heights_mm, confidence_scores)
    layer_metrics = analyze_layer_metrics(frame_heights_mm, layer_ids, confidence_scores)
    anomalies = detect_anomalies(frame_heights_mm, layer_ids)
    
    # Create and save visualization
    fig, axs = create_visualization(frame_heights_mm, frame_names, overlays, layer_ids, args)
    output_filename = get_output_filename(args)
    plt.savefig(output_filename)
    print(f"Plot saved to: {output_filename}")
    
    # Save metrics if requested
    all_metrics = {
        'video_name': os.path.basename(args.image_dir if args.image_dir else args.video if args.video else args.image),
        'timestamp': datetime.now().isoformat(),
        'expected_height_mm': args.expected_height_mm,
        'final_height_px': int(frame_heights[-1]),
        'scale_mm_per_px': scale_mm_per_px,
        'num_frames_processed': len(frame_heights),
        'num_layers_detected': num_layers,
        'temporal_metrics': temporal_metrics,
        'layer_metrics': layer_metrics,
        'anomalies': {k: len(v) for k, v in anomalies.items()}
    }
    save_metrics(all_metrics, args)
    
    # Show plot
    plt.show()
    
    # Clean up temporary files
    if tmp_dir and args.video:
        try:
            shutil.rmtree(tmp_dir)
            print(f"Temporary directory {tmp_dir} deleted.")
        except Exception as e:
            print(f"Failed to delete temporary directory {tmp_dir}: {e}")

if __name__ == "__main__":
    main()