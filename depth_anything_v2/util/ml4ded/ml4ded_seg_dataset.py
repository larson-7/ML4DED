import os
from glob import glob
from PIL import Image
import torch
import numpy as np
from depth_anything_v2.util.segbase import SegmentationDataset

class ML4DEDSegmentationDataset(SegmentationDataset):
    """
    ML4DED segmentation dataset (official_splits format):
        - Each split ('train', 'test') contains one folder per video.
        - Each video folder contains paired 'rgb_<video>_<frameid>.png' and 'seg_<video>_<frameid>.png'.

    Args:
        root (str): Path to the directory containing 'official_splits'.
        split (str): 'train' or 'test'.
        transform (callable, optional): Transform pipeline for RGB images.
        seg_transform (callable, optional): Transform pipeline for mask images.
        mode (str, optional): 'train' | 'val' | 'testval' (controls augmentation, if you subclass).
    """
    NUM_CLASS = 6
    IGNORE_LABEL = 255

    # 6‑class names in the ML4DED order
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
    ):
        super(ML4DEDSegmentationDataset, self).__init__(
            root, split, mode, transform
        )
        self.root = root
        self.split = split
        self.transform = transform
        self.seg_transform = seg_transform
        self.mode = mode

        split_dir = os.path.join(root, 'official_splits', split)
        # Recursively find all RGB image files matching 'rgb_*.png'
        rgb_paths = glob(os.path.join(split_dir, '**', 'rgb_*.png'), recursive=True)
        rgb_paths.sort()  # deterministic order

        self.files = []
        for rgb in rgb_paths:
            scene_dir, fname = os.path.split(rgb)
            idx = fname.replace('rgb_', '').replace('.png', '')      # e.g. '2_00001'
            mask = os.path.join(scene_dir, f"seg_{idx}.png")
            if not os.path.isfile(mask):
                raise FileNotFoundError(f"Missing mask for {rgb}")
            self.files.append((rgb, mask, idx))

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        print(f"[ML4DEDSegmentation] {split}: {len(self.files)} images")

    def __getitem__(self, index):
        rgb_path, mask_path, idx = self.files[index]

        img = np.array(Image.open(rgb_path).convert('RGB'))  # Convert to numpy array
        mask_img = Image.open(mask_path)

        if mask_img.size != (img.shape[1], img.shape[0]):
            mask_img = mask_img.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        mask = np.array(mask_img)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        mask = mask.long()
        return img, mask, idx

    def __len__(self):
        return len(self.files)

    def _mask_transform(self, mask):
        return mask  # PIL if torchvision transform is applied later

    def _img_transform(self, img):
        return img  # keep as PIL for torchvision

    @property
    def classes(self):
        return self._CLASSES



if __name__ == '__main__':
    ML4DEDSegmentationDataset()
