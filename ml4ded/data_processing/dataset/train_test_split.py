import os
import shutil
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets')
    parser.add_argument('--data-dir', type=str, default='',
                        help='Root directory containing data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    return parser.parse_args()

def copy_pairs(pairs, split, img_dir, mask_dir, split_dir):
    for video, frameid in pairs:
        split_video_dir = os.path.join(split_dir, split, video)
        os.makedirs(split_video_dir, exist_ok=True)
        img_src = os.path.join(img_dir, f'rgb_{video}_{frameid}.png')
        mask_src = os.path.join(mask_dir, f'seg_{video}_{frameid}.png')
        img_dst = os.path.join(split_video_dir, f'rgb_{video}_{frameid}.png')
        mask_dst = os.path.join(split_video_dir, f'seg_{video}_{frameid}.png')
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)
        if os.path.exists(mask_src):
            shutil.copy(mask_src, mask_dst)

def split_dataset(args):
    ROOT = args.root
    IMG_DIR = os.path.join(ROOT, 'segmentation_images')
    MASK_DIR = os.path.join(ROOT, 'segmentation_masks')
    SPLIT_DIR = os.path.join(ROOT, 'official_splits')

    # Clean out old splits if needed
    for split in ['train', 'test']:
        split_dir = os.path.join(SPLIT_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # Step 1: Group frame ids by video prefix using your parsing
    all_basenames = [f for f in os.listdir(IMG_DIR) if f.endswith('.png') and '_' in f]
    video_to_frameids = {}
    pairs = []
    for fname in all_basenames:
        type_, video, frameid = fname.split('_', 2)
        frameid = frameid.replace(".png", "")
        video_to_frameids.setdefault(video, []).append(frameid)
        pairs.append((video, frameid))

    # Step 2: Shuffle and IID split at frame level
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(args.train_ratio * len(pairs))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    print(f"IID split done with {args.train_ratio:.0%}/{1-args.train_ratio:.0%} train/test ratio!")
    print(f"Each train/test/[video]/ folder contains rgb_... and seg_... files (frame-level IID split).")

    copy_pairs(train_pairs, 'train', IMG_DIR, MASK_DIR, SPLIT_DIR)
    copy_pairs(test_pairs, 'test', IMG_DIR, MASK_DIR, SPLIT_DIR)

if __name__ == "__main__":
    args = parse_args()
    split_dataset(args)