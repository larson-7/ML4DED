import os
import json
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt

DATA_DIR = '/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded'
MASKS_DIR = os.path.join(DATA_DIR, 'layer_masks')
VIDEOS_DIR = os.path.join(DATA_DIR, 'layer_videos')
OUTPUT_IMAGES = os.path.join(DATA_DIR, 'segmentation_images')
OUTPUT_MASKS = os.path.join(DATA_DIR, 'segmentation_masks')
OUTPUT_MASKS_COLOR = os.path.join(DATA_DIR, 'segmentation_masks_color')

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)
os.makedirs(OUTPUT_MASKS_COLOR, exist_ok=True)

def get_label_colormap(num_labels):
    # Use matplotlib's new recommended interface (Matplotlib >=3.7)
    cmap = (plt.colormaps['tab20'](np.arange(num_labels))[:, :3] * 255).astype(np.uint8)
    cmap[0] = [0,0,0]  # background is black
    return cmap

# 1. Collect all unique object names globally
object_name_to_global_id = OrderedDict()
for subdir in sorted(os.listdir(MASKS_DIR)):
    subdir_path = os.path.join(MASKS_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue
    names_json = os.path.join(subdir_path, 'names.json')
    if not os.path.exists(names_json):
        continue
    with open(names_json, 'r') as f:
        names_map = json.load(f)
    for obj_id, obj_name in names_map.items():
        if obj_name not in object_name_to_global_id:
            object_name_to_global_id[obj_name] = len(object_name_to_global_id) + 1 # offset: 0 is background

global_id_to_name = {v: k for k, v in object_name_to_global_id.items()}
print(f"Global class mapping:\n{object_name_to_global_id}")

colormap = get_label_colormap(len(object_name_to_global_id) + 1)  # +1 for background

# 2. Process each mask dir
for subdir in sorted(os.listdir(MASKS_DIR)):
    mask_subdir = os.path.join(MASKS_DIR, subdir)
    if not os.path.isdir(mask_subdir):
        continue
    masks_json_path = os.path.join(mask_subdir, 'masks.json')
    names_json_path = os.path.join(mask_subdir, 'names.json')
    video_file = os.path.join(VIDEOS_DIR, f'{subdir}.mp4')
    if not all(map(os.path.exists, [masks_json_path, names_json_path, video_file])):
        print(f"Skipping {subdir}, missing files.")
        continue

    with open(names_json_path, 'r') as f:
        local_names = json.load(f)
    # Build local mapping: object_id -> global_id
    local_to_global = {}
    for str_obj_id, obj_name in local_names.items():
        local_to_global[int(str_obj_id)] = object_name_to_global_id[obj_name]

    with open(masks_json_path, 'r') as f:
        masks_data = json.load(f)

    # Decode all annotated masks into an array
    mask_frame_indices = sorted([int(k) for k in masks_data.keys()])
    decoded_masks = []
    for frame_idx in mask_frame_indices:
        mask_info = masks_data[str(frame_idx)]
        height, width = mask_info['results'][0]['mask']['size']
        mask_img = np.zeros((height, width), dtype=np.uint8)
        for obj in mask_info['results']:
            obj_id = obj['object_id']
            global_id = local_to_global.get(obj_id, 0)
            rle = {
                "counts": obj['mask']['counts'].encode('utf-8'),
                "size": obj['mask']['size']
            }
            decoded_mask = mask_utils.decode(rle)
            mask_img[decoded_mask == 1] = global_id
        decoded_masks.append(mask_img)

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open {video_file}")
        continue

    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    NUM_SAMPLES = min(10, num_video_frames)
    sample_indices = np.linspace(0, num_video_frames - 1, NUM_SAMPLES, dtype=int)
    sample_indices_set = set(sample_indices.tolist())

    for vframe_idx in tqdm(range(num_video_frames), desc=f'Processing {subdir}'):
        # Find bounding annotated frames
        if vframe_idx <= mask_frame_indices[0]:
            nearest_idx = 0
            interp_mask = decoded_masks[nearest_idx]
        elif vframe_idx >= mask_frame_indices[-1]:
            nearest_idx = -1
            interp_mask = decoded_masks[nearest_idx]
        else:
            for i in range(len(mask_frame_indices)-1):
                left_idx = mask_frame_indices[i]
                right_idx = mask_frame_indices[i+1]
                if left_idx <= vframe_idx <= right_idx:
                    if (vframe_idx - left_idx) <= (right_idx - vframe_idx):
                        interp_mask = decoded_masks[i]
                    else:
                        interp_mask = decoded_masks[i+1]
                    break

        # Save mask image
        mask_out_path = os.path.join(OUTPUT_MASKS, f'{subdir}_{vframe_idx:05d}.png')
        cv2.imwrite(mask_out_path, interp_mask.astype(np.uint8))

        # Save RGB frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, vframe_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {vframe_idx} from {video_file}")
            continue
        img_out_path = os.path.join(OUTPUT_IMAGES, f'{subdir}_{vframe_idx:05d}.png')
        cv2.imwrite(img_out_path, frame)

        # --- Color visualization for inspection ---
        if vframe_idx in sample_indices_set:
            color_mask = colormap[interp_mask]  # (H, W, 3)
            color_mask_bgr = color_mask[..., ::-1]  # OpenCV uses BGR

            # Resize color_mask to match frame, if needed
            if color_mask_bgr.shape[:2] != frame.shape[:2]:
                color_mask_bgr = cv2.resize(
                    color_mask_bgr,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            # Ensure frame is 3-channel
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(frame, 0.6, color_mask_bgr, 0.4, 0)
            color_path = os.path.join(OUTPUT_MASKS_COLOR, f'{subdir}_{vframe_idx:05d}_maskcolor.png')
            overlay_path = os.path.join(OUTPUT_MASKS_COLOR, f'{subdir}_{vframe_idx:05d}_overlay.png')
            cv2.imwrite(color_path, color_mask_bgr)
            cv2.imwrite(overlay_path, overlay)

    cap.release()
