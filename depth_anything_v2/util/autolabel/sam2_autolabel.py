import os
import glob
from PIL import Image
import torch
from sam2 import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np


def get_sam2_model(device="cuda"):
    # Download weights or point to local path
    checkpoint = "sam2_hiera_large.pth"  # Download from HuggingFace if needed
    model_type = "hiera-large"  # Options: "hiera-large", "vit-huge"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return sam


def masks_to_seg_map(masks, image_shape):
    # Each pixel gets the label of the highest scoring mask it belongs to.
    seg_map = np.zeros(image_shape[:2], dtype=np.uint8)
    for i, mask in enumerate(sorted(masks, key=lambda m: m['area'], reverse=True)):
        seg_map[mask['segmentation']] = i + 1  # 0=background, 1...N = objects
    return seg_map


def save_mask(mask, ref_path):
    mask_img = Image.fromarray(mask)
    save_path = os.path.join(os.path.dirname(ref_path), "seg_" + os.path.basename(ref_path))
    mask_img.save(save_path)


def main(root_dir, device="cuda"):
    # Load SAM2 model and mask generator
    sam = get_sam2_model(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    img_paths = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
    print(f"Found {len(img_paths)} .jpg files")

    for img_path in img_paths:
        print(f"Processing: {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        masks = mask_generator.generate(img)
        seg_map = masks_to_seg_map(masks, img.shape)
        save_mask(seg_map, img_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <directory> [cuda/cpu]")
        exit(1)
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    main(sys.argv[1], device)
