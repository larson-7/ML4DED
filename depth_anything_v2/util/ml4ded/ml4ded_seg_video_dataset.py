import os
from glob import glob
from PIL import Image
import torch
import numpy as np
from depth_anything_v2.util.segbase import SegmentationDataset

class ML4DEDSegmentationDataset(SegmentationDataset):
    """
    ML4DED segmentation dataset with support for temporal sequences.
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
        sequence_length=1  # Number of frames in a temporal sequence
    ):
        super(ML4DEDSegmentationDataset, self).__init__(
            root, split, mode, transform
        )
        self.root = root
        self.split = split
        self.transform = transform
        self.seg_transform = seg_transform
        self.mode = mode
        self.sequence_length = sequence_length

        split_dir = os.path.join(root, 'official_splits', split)
        rgb_paths = glob(os.path.join(split_dir, '**', 'rgb_*.png'), recursive=True)
        rgb_paths.sort()

        # Group samples by video ID
        self.video_to_samples = {}
        for rgb in rgb_paths:
            scene_dir, fname = os.path.split(rgb)
            idx = fname.replace('rgb_', '').replace('.png', '')  # e.g., '3_00017'
            mask = os.path.join(scene_dir, f"seg_{idx}.png")
            if not os.path.isfile(mask):
                raise FileNotFoundError(f"Missing mask for {rgb}")
            video_id = idx.split('_')[0]
            self.video_to_samples.setdefault(video_id, []).append((rgb, mask, idx))

        # Filter to only samples that can return full sequences
        self.sequence_index = []
        for video_id, samples in self.video_to_samples.items():
            samples.sort(key=lambda x: x[2])  # sort by frame index (idx)
            for i in range(len(samples) - sequence_length + 1):
                self.sequence_index.append((video_id, i))

        print(f"[ML4DEDSegmentation] {split}: {len(self.sequence_index)} sequences (length={sequence_length})")

    def __getitem__(self, index):
        video_id, start_idx = self.sequence_index[index]
        samples = self.video_to_samples[video_id][start_idx:start_idx + self.sequence_length]

        imgs = []
        masks = []
        indices = []

        for rgb_path, mask_path, idx in samples:
            img = np.array(Image.open(rgb_path).convert('RGB'))
            mask_img = Image.open(mask_path)

            if mask_img.size != (img.shape[1], img.shape[0]):
                mask_img = mask_img.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
            mask = np.array(mask_img)

            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            imgs.append(img)
            masks.append(mask.long())
            indices.append(idx)

        if self.sequence_length == 1:
            return imgs[0], masks[0], indices[0]
        return imgs, masks, indices  # list of tensors and list of idxs

    def __len__(self):
        return len(self.sequence_index)

    @property
    def classes(self):
        return self._CLASSES
