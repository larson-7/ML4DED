#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Extract RGB, depth‑in‑mm, and 40‑class segmentation maps
#   from nyu_depth_v2_labeled.mat  (or its HDF5 conversion).
#
#   Example Usage:
#   python extract_nyu_depth_and_seg40.py \
#          nyu_depth_v2_labeled.mat  splits.mat  labels40.mat  /out/dir
#
#   Output tree (matches BTS / EVP expectations):
#       /out/dir/{train,test}/{scene}/
#               rgb_00000.jpg
#               sync_depth_00000.png
#               seg40_00000.png

#   The seg PNG is uint8 with values 0‑39. Pixels whose raw label maps to
#   class 0 in NYU’s 40‑class definition are unlikely / void – feel free
#   to treat 0 or 255 as ignore‑index in your loss.
# ----------------------------------------------------------------------

from __future__ import print_function

import h5py
import numpy as np
import os
import scipy.io
import sys
import cv2


def convert_image(i, scene, depth_raw, image, label_raw):
    idx = i + 1  # 1‑based in splits.mat

    train_test = "train" if idx in train_images else "test"
    folder = f"{out_folder}/{train_test}/{scene}"
    os.makedirs(folder, exist_ok=True)

    # ------- depth (same as original script) --------------------------
    img_depth = (depth_raw * 1000.0).astype(np.uint16)   # metres → mm
    cv2.imwrite(f"{folder}/sync_depth_{i:05d}.png", img_depth)

    # ------- RGB (7‑pixel crop to match NYU protocol) ------------------
    image = image[:, :, ::-1]  # BGR for cv2
    rgb_out = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_out[7:474, 7:632] = image[7:474, 7:632]
    cv2.imwrite(f"{folder}/rgb_{i:05d}.jpg", rgb_out)

    # ------- segmentation (40‑class, same crop) -----------------------
    seg_out = np.full((480, 640), 255, dtype=np.uint8)  # init to void
    seg_out[7:474, 7:632] = label_raw[7:474, 7:632]
    cv2.imwrite(f"{folder}/seg40_{i:05d}.png", seg_out)


# ----------------------------------------------------------------------#
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"usage: {sys.argv[0]} <labeled_mat> <splits.mat> <labels40.mat> <out_dir>",
              file=sys.stderr)
        sys.exit(0)

    mat_path, split_path, map40_path, out_folder = sys.argv[1:5]

    # --------- load MATLAB .mat ---
    print(f"Loading labeled dataset   : {mat_path}")
    mat = h5py.File(mat_path, "r")
    depth_raw_all = mat['depths']       # (H,W,N)
    images_all    = mat['images']       # (H,W,3,N)
    labels_raw_all = mat['labels']      # (H,W,N)

    print(f"Loading train/test split  : {split_path}")
    split = scipy.io.loadmat(split_path)
    test_images  = set(int(x) for x in split["testNdxs"].squeeze())
    train_images = set(int(x) for x in split["trainNdxs"].squeeze())
    print(f"{len(train_images)} training images, {len(test_images)} test images")


    # scenes are MATLAB cell array of strings
    scenes = ["".join(chr(c[0]) for c in mat[ref][:]) for ref in mat['sceneTypes'][0]]


    print("Processing images...")
    for i in range(images_all.shape[0]):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{images_all.shape[0]}")
        convert_image(
            i,
            scenes[i],
            depth_raw_all[i, :, :].T,
            images_all[i, :, :, :].T,
            labels_raw_all[i, :, :].T,
        )
