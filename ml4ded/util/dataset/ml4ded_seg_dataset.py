import os
from glob import glob
from PIL import Image
import torch
import numpy as np
from collections import defaultdict
import albumentations as A

from ml4ded.util.dataset.segbase import SegmentationDataset


class ML4DEDSegmentationDataset(SegmentationDataset):
    """
    ML4DED segmentation dataset with temporal context.

    Args:
        root (str): Path to data root.
        split (str): 'train' or 'test'
        transform (callable): Data augmentations for both RGB and mask.
        seg_transform (callable): (Optional) Mask-only transform.
        mode (str): 'train', 'val', etc.
        temporal_window (int): Number of past frames to include (returns N+1 total frames).
    """
    NUM_CLASS = 6
    IGNORE_LABEL = 255
    _CLASSES = (
        "background",
        "head",
        "baseplate",
        "previous_weld",
        "weld",
        "current_part",
    )

    def __init__(
        self,
        root="/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded",
        split='train',
        transform=None,
        seg_transform=None,
        mode='train',
        temporal_window=0,
    ):
        super().__init__(root, split, mode, transform)
        self.root = root
        self.split = split
        self.transform = transform
        self.seg_transform = seg_transform
        self.mode = mode
        self.temporal_window = temporal_window

        split_dir = os.path.join(root, 'official_splits', split)
        rgb_paths = glob(os.path.join(split_dir, '**', 'rgb_*.png'), recursive=True)
        rgb_paths.sort()

        self.files = []
        self.index_to_context = {}

        # --- Group by video ID ---
        video_to_frames = defaultdict(list)

        for rgb_path in rgb_paths:
            scene_dir, fname = os.path.split(rgb_path)
            idx = fname.replace('rgb_', '').replace('.png', '')
            mask_path = os.path.join(scene_dir, f"seg_{idx}.png")
            if not os.path.isfile(mask_path):
                continue
            video_id, frame_num = self._parse_frame_info(fname)
            video_to_frames[video_id].append((frame_num, rgb_path, mask_path, idx))

        # --- Sort each video and build index/context mapping ---
        for video_id, frames in video_to_frames.items():
            frames.sort(key=lambda x: x[0])  # sort by frame_num
            for i in range(len(frames)):
                context_indices = list(range(max(0, i - self.temporal_window), i + 1))
                context = [frames[j][1:] for j in context_indices]  # (rgb_path, mask_path, idx)
                self.index_to_context[len(self.files)] = context
                self.files.append(frames[i][1:])  # only append the current frame entry

        print(f"[ML4DEDSegmentation] {split}: {len(self.files)} frames with up to {temporal_window} previous frames")

    def _parse_frame_info(self, filename):
        """
        Example: 'rgb_2_00003.png' → ('2', 3)
        """
        name = os.path.splitext(filename)[0]  # 'rgb_2_00003'
        parts = name.split('_')  # ['rgb', '2', '00003']
        return parts[1], int(parts[2])

    def __getitem__(self, index):
        """
        Returns:
            img_seq: Tensor of shape (T, 3, H, W)
            mask_seq: Tensor of shape (T, H, W)
            idx: str — identifier for the most recent (target) frame
        """
        # Retrieve context list — ensure full temporal window is generated (with padding if needed)
        target_context = self.index_to_context[index]  # this has only the target frame info
        video_id, frame_num = self._parse_frame_info(target_context[-1][0])  # Get info from latest frame

        # Get all frames for the video and sort them
        video_frames = sorted([
            entry for entry in self.index_to_context.values()
            if self._parse_frame_info(entry[-1][0])[0] == video_id
        ], key=lambda x: self._parse_frame_info(x[-1][0])[1])

        # Find current frame's index in sorted video frames
        current_idx = next(i for i, ctx in enumerate(video_frames)
                           if ctx[-1][0] == target_context[-1][0])

        # Gather temporal window with padding (repeat first frame if needed)
        padded_context = []
        for j in range(current_idx - self.temporal_window, current_idx + 1):
            if j < 0:
                padded_context.append(video_frames[0][-1])  # Use first frame
            else:
                padded_context.append(video_frames[j][-1])  # Use actual frame

        imgs, masks, idxs = [], [], []

        for rgb_path, mask_path, idx in padded_context:
            img = np.array(Image.open(rgb_path).convert('RGB'))
            mask_img = Image.open(mask_path)

            if mask_img.size != (img.shape[1], img.shape[0]):
                mask_img = mask_img.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)

            mask = np.array(mask_img)
            imgs.append(img)
            masks.append(mask)
            idxs.append(idx)

        # Apply consistent augmentation if transform is ReplayCompose
        if self.transform is not None:
            if isinstance(self.transform, A.ReplayCompose):
                first_aug = self.transform(image=imgs[0], mask=masks[0])
                replay = first_aug['replay']
                imgs_aug = [first_aug['image']]
                masks_aug = [first_aug['mask'].long()]
                for img, mask in zip(imgs[1:], masks[1:]):
                    aug = A.ReplayCompose.replay(replay, image=img, mask=mask)
                    imgs_aug.append(aug['image'])
                    masks_aug.append(aug['mask'].long())
            else:
                imgs_aug, masks_aug = [], []
                for img, mask in zip(imgs, masks):
                    aug = self.transform(image=img, mask=mask)
                    imgs_aug.append(aug['image'])
                    masks_aug.append(aug['mask'].long())

            img_seq = torch.stack(imgs_aug, dim=0)  # (T, 3, H, W)
            mask_seq = torch.stack(masks_aug, dim=0)  # (T, H, W)
        else:
            raise ValueError("Transform must be defined for temporal mode")

        return img_seq, mask_seq, idxs[-1]

    def __len__(self):
        return len(self.files)

    def _mask_transform(self, mask):
        return mask

    def _img_transform(self, img):
        return img

    @property
    def classes(self):
        return self._CLASSES
