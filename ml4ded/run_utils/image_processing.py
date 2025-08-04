import time
import os
import numpy as np
import torch

from PIL import Image
from scipy.ndimage import median_filter, label
from ml4ded.data_processing.vis import decode_segmap

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
        
    if conf_map is not None:
        # Only keep pixels where confidence exceeds threshold
        valid_mask = mask & (conf_map > threshold)
        if not np.any(valid_mask):
            print(f"Warning: No pixels with confidence > {threshold} for class {target_class}")
            return None
        mask = valid_mask
    
    labeled_mask, num_components = label(mask)
    if num_components == 0:
        return None
        
    component_sizes = np.bincount(labeled_mask.ravel())[1:] if num_components > 0 else []
    if len(component_sizes) == 0:
        return None
        
    largest_component_label = np.argmax(component_sizes) + 1
    largest_component_size = component_sizes[largest_component_label-1]
    
    # Filter out small components
    total_area = np.sum(mask)
    if largest_component_size < (min_area_pct * total_area) and num_components > 1:
        print(f"Warning: Largest component too small ({largest_component_size}/{total_area} pixels)")
        return None
        
    # Create a mask with only the largest component
    main_component_mask = (labeled_mask == largest_component_label)
    
    # Calculate height from largest component only
    ys = np.where(main_component_mask)[0]
    height_px = ys.max() - ys.min() + 1
    return height_px

def overlay_segmentation(image_np, seg_map, alpha=0.5):
    seg_color = decode_segmap(seg_map)
    overlay = (alpha * seg_color + (1 - alpha) * image_np).astype(np.uint8)
    return overlay

def process_frames(model, input_transform, frames, args, target_class, image_dims):
    """Process frames through the model and compute heights."""
    frame_heights = []
    frame_names = []
    overlays = []
    confidence_scores = []
    num_frames = len(frames)
    device = args.device
    img_w, img_h = image_dims
    
    for i, entry in enumerate(frames, 1):
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
        mask = (seg_map_np[0] == target_class)
        if np.any(mask):
            avg_confidence = np.mean(confidence_map[mask])
            max_confidence = np.max(confidence_map[mask])
            num_pixels = np.sum(mask)
            print(f"CURRENT_PART confidence - avg: {avg_confidence:.3f}, max: {max_confidence:.3f}, pixels: {num_pixels}")
            confidence_scores.append(avg_confidence)
        else:
            confidence_scores.append(0.0)

        # Compute object height
        height_px = compute_object_height(
            seg_map_np, 
            target_class, 
            conf_map=confidence_map, 
            threshold=args.confidence_threshold,
            min_area_pct=args.min_area_pct if args.enable_temporal else 0.0
        )

        if height_px is None:
            print(f"{os.path.basename(entry)}: No CURRENT_PART detected. Skipping.")
            continue

        frame_names.append(entry)
        frame_heights.append(height_px)

        # Create overlay image
        img_np = np.array(raw_img.resize((img_w, img_h)))
        filtered_seg_map = seg_map_np.copy()
        
        if args.show_main_component:
            mask = (seg_map_np[0] == target_class)
            if np.any(mask):
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
            
            overlay_img = overlay_segmentation(img_np, filtered_seg_map, alpha=0.5)
        else:
            # Standard segmentation overlay with all components
            overlay_img = overlay_segmentation(img_np, seg_map_np, alpha=0.5)
            
        overlays.append(overlay_img)
    
    return frame_heights, frame_names, overlays, confidence_scores


def apply_filtering(frame_heights, args):
    """Apply filters to height measurements."""
    # Store original heights before any filtering
    original_heights = np.array(frame_heights).copy()

    if len(frame_heights) > 1 and args.use_median:
        window_size = args.filter_window
        if window_size % 2 == 0:  # Ensure window size is odd
            window_size += 1
        print(f"Applying median filter with window size {window_size}...")
        filtered_heights = median_filter(frame_heights, size=window_size)
        
        # Print stats about filtering
        changes = np.abs(original_heights - filtered_heights)
        if np.any(changes > 0):
            avg_change = np.mean(changes[changes > 0])
            max_change = np.max(changes)
            num_changed = np.sum(changes > 0)
            print(f"Filtering changed {num_changed}/{len(frame_heights)} values")
            print(f"Average change: {avg_change:.2f}px, Max change: {max_change:.2f}px")
        
        return filtered_heights, original_heights
    
    return np.array(frame_heights), original_heights