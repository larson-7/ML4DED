import os
import shutil
import random

ROOT = '/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded'
IMG_DIR = os.path.join(ROOT, 'segmentation_images')
MASK_DIR = os.path.join(ROOT, 'segmentation_masks')
SPLIT_DIR = os.path.join(ROOT, 'official_splits')

# Clean old splits
for split in ['train', 'test']:
    split_dir = os.path.join(SPLIT_DIR, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

# Step 1: Group all files by video ID
video_to_frames = {}
for fname in os.listdir(IMG_DIR):
    if not fname.endswith('.png') or not fname.startswith('rgb_'):
        continue
    parts = fname.replace(".png", "").split('_')
    _, video, frameid = parts
    video_to_frames.setdefault(video, []).append(frameid)

# Step 2: Temporal split (at video level)
video_ids = sorted(video_to_frames.keys())
random.seed(42)
random.shuffle(video_ids)
split_idx = int(0.8 * len(video_ids))
train_videos = video_ids[:split_idx]
test_videos = video_ids[split_idx:]

# Step 3: Copy files for each split
def copy_video_sequences(video_ids, split):
    for video in video_ids:
        frames = video_to_frames[video]
        split_video_dir = os.path.join(SPLIT_DIR, split, video)
        os.makedirs(split_video_dir, exist_ok=True)
        for frameid in frames:
            img_src = os.path.join(IMG_DIR, f'rgb_{video}_{frameid}.png')
            mask_src = os.path.join(MASK_DIR, f'seg_{video}_{frameid}.png')
            img_dst = os.path.join(split_video_dir, f'rgb_{video}_{frameid}.png')
            mask_dst = os.path.join(split_video_dir, f'seg_{video}_{frameid}.png')
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
            if os.path.exists(mask_src):
                shutil.copy(mask_src, mask_dst)

copy_video_sequences(train_videos, 'train')
copy_video_sequences(test_videos, 'test')

print("Temporal split done! Each train/test/[video]/ folder contains full sequences.")
