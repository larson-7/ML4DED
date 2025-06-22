import argparse
import os
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from enum import Enum
from time import time

import open3d as o3d  # <--- The real 3D viewer!

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
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Depth Inference')
    parser.add_argument('--data-dir', type=str, default="../data/ml4ded/official_splits",
                        help='train/test data directory')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights",
                        help='pretrained model weights directory')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--class-a', type=int, default=SegLabels.BASEPLATE.value, help='Class A ID for relative depth comparison')
    parser.add_argument('--class-b', type=int, default=SegLabels.CURRENT_PART.value, help='Class B ID for relative depth comparison')
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def plot_panel_results(raw_img, seg_map_color, depth_map_vis):
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

def show_vggt_3d(rgb_img, depth_map, seg_map, class_id, max_points=50000):
    """
    Shows interactive Open3D window of the class mask as colored point cloud (VGG-T style!)
    """
    mask = seg_map == class_id
    if not np.any(mask):
        print(f"No pixels found for class {class_id}")
        return

    # Masked coordinates and color
    ys, xs = np.where(mask)
    zs = depth_map[mask]
    img_arr = np.array(rgb_img)
    colors = img_arr[mask].astype(np.float32) / 255.0

    # Subsample for speed/clarity if too large
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), max_points, replace=False)
        xs, ys, zs, colors = xs[idx], ys[idx], zs[idx], colors[idx]

    # (N, 3) points in image space (X, Y, Z)
    points = np.stack([xs, ys, zs], axis=1)
    # You may want to flip the y axis to match top-left image origin
    points[:, 1] = rgb_img.height - points[:, 1]

    # Build open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Launching interactive 3D viewer (close the window to continue)...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"VGG-T Style 3D: {class_id}",
        width=1280,
        height=960,
        point_show_normal=False
    )

def local_rim_depth_diff(seg_map, depth, class_A, class_B, SegLabels, rim_width=5):
    mask_A = (seg_map == class_A)
    mask_B = (seg_map == class_B)

    if not (np.any(mask_A) and np.any(mask_B)):
        print(f"\nCannot compute relative depth: one or both classes not present in segmentation.")
        return

    mask_B_2d = mask_B
    mask_A_2d = mask_A
    depth_2d = depth

    # Dilate mask B to create a rim region around B
    dilated_B = binary_dilation(mask_B_2d, iterations=rim_width)
    rim_A = np.logical_and(dilated_B, mask_A_2d)

    # Get coordinates
    by, bx = np.where(mask_B_2d)
    ay, ax = np.where(rim_A)
    if len(ay) == 0:
        print("No rim found, try increasing rim_width.")
        return

    # Vectorized: match each B pixel to its *nearest* rim A pixel
    A_coords = np.stack([ay, ax], axis=1)
    B_coords = np.stack([by, bx], axis=1)

    if len(B_coords) > 1000:
        idx = np.random.choice(len(B_coords), 1000, replace=False)
        B_coords = B_coords[idx]

    dists = np.linalg.norm(B_coords[:, None, :] - A_coords[None, :, :], axis=2)
    idx_closest = np.argmin(dists, axis=1)
    A_closest = A_coords[idx_closest]

    d_B = depth_2d[B_coords[:, 0], B_coords[:, 1]]
    d_A = depth_2d[A_closest[:, 0], A_closest[:, 1]]

    avg_diff = np.mean(d_B - d_A)
    print(f"\n--- Relative Depth (rim method) ---")
    print(f"Avg. Depth ({SegLabels(class_B).name}) - Local Rim Depth ({SegLabels(class_A).name}) = {avg_diff:.4f}")

if __name__ == '__main__':
    args = parse_args()
    raw_img = Image.open(args.image_path).convert('RGB')
    img_w, img_h = make_divisible(raw_img.size)
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

    raw_img_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
    ])

    transformed_image = input_transform(raw_img).unsqueeze(0).to(args.device)
    start_time = time()
    with torch.no_grad():
        depth, seg_map = model.infer_image(transformed_image)
    end_time = time()
    print(f"SegDepthInference took {end_time - start_time:.2f} seconds")

    # Process segmentation output
    seg_map = seg_map[0]
    seg_map_color = decode_segmap(seg_map)  # (H, W, 3)

    # Process depth output
    depth_map = depth[0]
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map_inv = 1.0 - depth_map_norm
    depth_map_vis = cv2.applyColorMap((depth_map_inv * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    cropped_raw_img = raw_img_transform(raw_img)
    # Show classic panel for quick overview
    plot_panel_results(
        cropped_raw_img,
        seg_map_color,
        depth_map_vis,
    )

    # --- VGG-T style interactive 3D for just the selected object
    show_vggt_3d(
        cropped_raw_img,
        depth_map,
        seg_map,
        class_id=args.class_b,  # Typically the object of interest
        max_points=100000
    )

    # ---- Metrics ----
    class_ids = np.unique(seg_map)
    print("\n--- Depth Metrics Per Class ---")
    for cls_id in class_ids:
        mask = (seg_map == cls_id)
        if mask.sum() < 10:
            continue
        class_depths = depth_map[mask]
        mean_depth = np.mean(class_depths)
        std_depth = np.std(class_depths)
        print(f"Class {cls_id} ({SegLabels(cls_id).name}): mean depth = {mean_depth:.4f}, std = {std_depth:.4f}")

        # Rank correlation
        y_coords, x_coords = np.where(mask)
        flat_positions = x_coords + y_coords * depth_map.shape[1]
        rho, _ = spearmanr(flat_positions, class_depths.flatten())
        print(f"   Spearman rank correlation ρ = {rho:.3f}")

    # --- Relative Depth: Class B vs Class A ---
    class_A = args.class_a
    class_B = args.class_b

    local_rim_depth_diff(seg_map, depth_map, class_A, class_B, SegLabels, rim_width=10)
