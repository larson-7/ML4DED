import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class ML4DEDSegmentationDataset(Dataset):
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
    IGNORE_LABEL = 255

    def __init__(
        self,
        root="/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded",
        split='train',
        transform=None,
        seg_transform=None,
        mode='train',
    ):
        super().__init__()
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

        img = Image.open(rgb_path).convert('RGB')
        mask = Image.open(mask_path)  # 'L', uint8

        # Optionally apply transforms (see SegmentationDataset logic in your codebase)
        if self.transform is not None:
            img = self.transform(img)
        if self.seg_transform is not None:
            mask = self.seg_transform(mask)
            # ToTensor rescales to [0,1]; restore 0-255 and long dtype for mask
            if isinstance(mask, torch.Tensor):
                mask = (mask * 255).long()

        return img, mask, idx

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ML4DEDSegmentationDataset()
