import argparse
import time
import datetime
import os
import shutil
import sys

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from depth_anything_v2.train import make_divisible

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from dino2seg import Dino2Seg, DPTSegmentationHead
from util.segmentationMetric import *
from util.vis import decode_segmap
from depth_anything_v2.dinov2 import DINOv2
from util.nyu_d_v2.nyudv2_seg_dataset import NYUSDv2SegDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--data-dir', type=str, default="../data/nyu_depth_v2",
                        help='train/test data directory')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights",
                        help='pretrained model weights directory')

    parser.add_argument('--base-size', type=int, default=580,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=518,
                        help='crop image size')

    # training hyper params
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')

    # checkpoint and log
    parser.add_argument('--save-dir', default='./ckpt',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--device', default='cuda',
                        help='Training device')
    args = parser.parse_args()
    return args


def make_divisible(val, divisor=14):
    return val - (val % divisor)

if __name__ == '__main__':
    args = parse_args()
    raw_img = Image.open("/home/jordan/omscs/cs8903/SegDefDepth/data/ml4ded/buildplate000_1/003800.jpg")
    img_w, img_h = make_divisible(np.array(raw_img.size))

    model = Dino2Seg(
        encoder="vitb",
        num_classes=41,
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=args.model_weights_dir,
    )

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    transformed_image = input_transform(raw_img)
    transformed_image = transformed_image.unsqueeze(0).to(args.device)
    outputs = model(transformed_image)
    pred_labels = torch.max(outputs, 1).indices
    pred = decode_segmap(pred_labels[0].cpu().data.numpy())
    print("end")

