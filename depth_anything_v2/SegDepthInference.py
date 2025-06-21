import argparse
import os
import sys
import numpy as np
from scipy.ndimage import label, find_objects
from PIL import Image
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from util.vis import decode_segmap
from time import time


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Depth Inference')
    parser.add_argument('--data-dir', type=str, default="../data/ml4ded/official_splits",
                        help='train/test data directory')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights",
                        help='pretrained model weights directory')
    parser.add_argument('--base-size', type=int, default=580, help='base image size')
    parser.add_argument('--crop-size', type=int, default=518, help='crop image size')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--save-dir', default='./ckpt', help='Directory for saving checkpoint models')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--class-a', type=int, default=2, help='Class A ID for relative depth comparison')
    parser.add_argument('--class-b', type=int, default=1, help='Class B ID for relative depth comparison')
    args = parser.parse_args()
    return args


def make_divisible(val, divisor=14):
    return val - (val % divisor)


if __name__ == '__main__':
    args = parse_args()
    raw_img = Image.open(args.image_path)
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
    model.eval().to(args.device)

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    transformed_image = input_transform(raw_img).unsqueeze(0).to(args.device)
    start_time = time()
    depth, seg_map = model.infer_image(transformed_image)
    end_time = time()
    print(f"SegDepthInference took {end_time - start_time} seconds")
    # Process segmentation output
    seg_map_color = decode_segmap(seg_map[0])  # (H, W, 3)

    # Process depth output
    depth_map = depth[0]
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map_inv = 1.0 - depth_map_norm
    depth_map_vis = cv2.applyColorMap((depth_map_inv * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(raw_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(seg_map_color)
    axs[1].set_title('Predicted Segmentation')
    axs[1].axis('off')
    axs[2].imshow(depth_map_vis)
    axs[2].set_title('Predicted Depth')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

    # ---- Metrics ----
    class_ids = np.unique(seg_map)
    print("\n--- Depth Metrics Per Class ---")
    for cls_id in class_ids:
        mask = (seg_map == cls_id)
        if mask.sum() < 10:
            continue
        class_depths = depth[mask]
        mean_depth = np.mean(class_depths)
        std_depth = np.std(class_depths)
        print(f"Class {cls_id}: mean depth = {mean_depth:.4f}, std = {std_depth:.4f}")

        # Rank correlation
        y_coords, x_coords = np.where(mask[0])
        flat_positions = x_coords + y_coords * depth.shape[1]
        rho, _ = spearmanr(flat_positions, class_depths.flatten())
        print(f"   Spearman rank correlation ρ = {rho:.3f}")

    # --- Relative Depth: Class B vs Class A ---
    class_A = args.class_a
    class_B = args.class_b
    mask_A = (seg_map == class_A)
    mask_B = (seg_map == class_B)

    if np.any(mask_A) and np.any(mask_B):
        coords_A = np.column_stack(np.where(mask_A[0]))
        coords_B = np.column_stack(np.where(mask_B[0]))
        min_diffs = []

        for ax, ay in coords_A:
            dists = np.linalg.norm(coords_B - np.array([ax, ay]), axis=1)
            idx_closest = np.argmin(dists)
            bx, by = coords_B[idx_closest]
            d_A = depth[0, ax, ay]
            d_B = depth[0, bx, by]
            min_diffs.append(d_B - d_A)

        if len(min_diffs) > 0:
            avg_diff = np.mean(min_diffs)
            print(f"\n--- Relative Depth ---")
            print(f"Avg. Depth (Class {class_B}) - Depth (Class {class_A}) = {avg_diff:.4f}")
    else:
        print(f"\nCannot compute relative depth: one or both classes not present in segmentation.")


def find_object_pixel_indices(seg_map: np.ndarray, target_label: int):
    """
    Finds pixel indices for each connected component (object) of the target label.

    Parameters:
    - seg_map: np.ndarray, shape (H, W), segmentation map
    - target_label: int, the label to extract connected objects for

    Returns:
    - object_indices: list of np.ndarrays, each array is shape (N, 2) with (row, col) of pixels for each object
    """
    # Binary mask where label matches
    mask = (seg_map == target_label).astype(np.uint8)

    # Label connected components in the mask
    labeled_array, num_features = label(mask)

    # Extract indices for each component
    object_indices = []
    for component_id in range(1, num_features + 1):
        rows, cols = np.where(labeled_array == component_id)
        indices = np.stack((rows, cols), axis=1)  # shape (N, 2)
        object_indices.append(indices)

    return object_indices


def find_relative_height(seg_map, depth_map, label_a, label_b, margin=5):
    """
    Measures relative depth between label_a objects and adjacent label_b regions.

    Parameters:
    - seg_map: np.ndarray of shape (H, W), semantic segmentation map
    - depth_map: np.ndarray of shape (H, W), depth map aligned to seg_map
    - label_a: int, object of interest
    - label_b: int, reference object (e.g., baseplate)
    - margin: int, number of pixels to expand A's bounding box

    Returns:
    - List of dicts with 'object_index', 'depth_a', 'depth_b', 'relative_depth'
    """
    results = []

    # Label connected components of object A
    mask_a = (seg_map == label_a).astype(np.uint8)
    labeled_a, num_a = label(mask_a)

    for obj_idx in range(1, num_a + 1):
        coords = np.array(np.where(labeled_a == obj_idx)).T
        if coords.size == 0:
            continue

        rows, cols = coords[:, 0], coords[:, 1]

        # Compute bounding box with margin
        rmin = max(rows.min() - margin, 0)
        rmax = min(rows.max() + margin + 1, seg_map.shape[0])
        cmin = max(cols.min() - margin, 0)
        cmax = min(cols.max() + margin + 1, seg_map.shape[1])

        # Mask for A and B in bounding box
        region = seg_map[rmin:rmax, cmin:cmax]
        depth_region = depth_map[rmin:rmax, cmin:cmax]

        mask_a_region = (region == label_a)
        mask_b_region = (region == label_b)

        # Extract depth values
        depth_a = depth_region[mask_a_region]
        depth_b = depth_region[mask_b_region]

        if len(depth_a) == 0 or len(depth_b) == 0:
            continue

        mean_a = np.median(depth_a)
        mean_b = np.median(depth_b)
        rel_depth = mean_b - mean_a

        results.append({
            "object_index": obj_idx,
            "depth_a": float(mean_a),
            "depth_b": float(mean_b),
            "relative_depth": float(rel_depth),
        })

    return results
