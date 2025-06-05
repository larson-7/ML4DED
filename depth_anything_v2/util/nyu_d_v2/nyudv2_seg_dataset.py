# nyu_official_seg.py
# ----------------------------------------------------------------------
import os
import torch
import numpy as np

from glob import glob
from PIL import Image
from depth_anything_v2.util.segbase import SegmentationDataset


class NYUSDv2SegDataset(SegmentationDataset):
    """
    NYU‑Depth‑v2 40‑class segmentation (795 train / 654 test) extracted by
    your 'official_splits' converter.

    • Expects rgb_* and seg40_* inside per‑scene sub‑dirs.
    • Mask values are 0‑39 (40 classes). 255 = void / ignore.


    Example Usage:
        from torchvision import transforms
        from torch.utils.data import DataLoader

        rgb_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        mask_tf = transforms.Compose([transforms.ToTensor()])  # optional

        train_ds = NYUSegmentationOfficialSplit(
            root='nyu_depth_v2',
            split='train',
            mode='train',
            transform=rgb_tf,
            seg_transform=mask_tf
        )

        val_ds   = NYUSegmentationOfficialSplit(
            root='nyu_depth_v2',
            split='test',
            mode='val',
            transform=rgb_tf,
            seg_transform=mask_tf
        )

        loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

        for imgs, masks, ids in loader:
            # imgs  : (B,3,H,W) float32
            # masks : (B,H,W)   long  (0‑39, 255=void)
    """

    NUM_CLASS = 40
    IGNORE_LABEL = 255

    # 40‑class names in the NYU order
    _CLASSES = (
        'bed', 'books', 'ceil', 'chair', 'floor', 'furn', 'objs', 'paint', 'sofa',
        'table', 'tv', 'wall', 'window', 'blinds', 'desk', 'shelves', 'door',
        'pillow', 'sink', 'lamp', 'tub', 'stove', 'toilet', 'curtain', 'dresser',
        'fridge', 'tv‑monitor', 'radiator', 'glass', 'whiteboard', 'person',
        'clothes', 'ceiling‑fan', 'plant', 'paper', 'towel', 'shower‑curtain',
        'box', 'board‑panel', 'bookshelf'
    )

    def __init__(
        self,
        root='../../../data/nyu_depth_v2',   # folder holding official_splits/
        split='train',                     # 'train' or 'test'
        mode=None,                         # 'train' | 'val' | 'testval'
        transform=None,                    # transform pipeline for RGB
        seg_transform=None                 # optional separate transform for mask
    ):
        super(NYUSDv2SegDataset, self).__init__(
            root, split, mode, transform
        )
        self.seg_transform = seg_transform
        assert split in ('train', 'test'), "split must be 'train' or 'test'"

        # ------------------------------------------------------------------
        # Scan all scene sub‑folders and build a list of (rgb_path, mask_path)
        split_dir = os.path.join(root, 'official_splits', split)
        # Recursively find all RGB image files matching 'rgb_*.jpg'
        rgb_paths = glob(os.path.join(split_dir, '**', 'rgb_*.jpg'), recursive=True)
        rgb_paths.sort()                              # deterministic order

        self.files = []
        for rgb in rgb_paths:
            scene_dir, fname = os.path.split(rgb)
            idx = fname.replace('rgb_', '').split('.')[0]          # 00042
            mask = os.path.join(scene_dir, f"seg40_{idx}.png")     # or .jpg
            if not os.path.isfile(mask):
                raise FileNotFoundError(f"Missing mask for {rgb}")
            self.files.append((rgb, mask, idx))

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        print(f"[NYUSegmentation] {split}: {len(self.files)} images")

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        rgb_path, mask_path, idx = self.files[index]

        img = Image.open(rgb_path).convert('RGB')
        mask = Image.open(mask_path)                    # 'L', uint8

        # synched transforms (from SegmentationDataset base)
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:  # 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.seg_transform is not None:
            mask = self.seg_transform(mask)
            # ToTensor() rescales to [0,1] → restore 0‑255 then long
            if isinstance(mask, torch.Tensor):
                mask = (mask * 255).long()

        return img, mask, idx

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.files)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask, dtype='uint8')).long()

    @property
    def classes(self):
        return self._CLASSES


if __name__ == '__main__':
    NYUSDv2SegDataset()