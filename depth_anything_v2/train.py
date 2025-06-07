import argparse
import time
import datetime
import os
import shutil
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

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
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # NYUv2: original size is 640×480 → crop to 630×476 for compatibility
        img_h, img_w = make_divisible(480), make_divisible(640)

        # image transform (normalize to imagenet mean statistics)
        input_transform = transforms.Compose([
            transforms.CenterCrop((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        seg_transform = transforms.Compose([
            transforms.CenterCrop((img_h, img_w)),
            transforms.ToTensor()])

        # dataset and dataloader
        trainset = NYUSDv2SegDataset(args.data_dir, split="train", mode="train", transform=input_transform, seg_transform=seg_transform)
        valset = NYUSDv2SegDataset(args.data_dir, split="test", mode="val", transform=input_transform, seg_transform=seg_transform)

        self.train_loader = data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=valset, batch_size=args.batch_size,
                                          pin_memory=True)

        self.model = Dino2Seg(
            encoder="vitb",
            num_classes=len(trainset.classes),
            image_height=img_h,
            image_width=img_w,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            model_weights_dir=args.model_weights_dir,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=args.lr)

        self.metric = SegmentationMetric(len(trainset.classes))
        self.best_pred = -1

    def train(self):
        iteration = 0
        avg_loss = 0
        for i in range(args.epochs):
            self.validation(iteration, i)
            self.model.train()
            for images, targets, _ in self.train_loader:
                iteration = iteration + 1
                # self.lr_scheduler.step()

                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                pred = torch.max(outputs, 1).indices
                loss = self.criterion(outputs, targets)

                loss = torch.mean(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss

                if iteration % 100 == 0:
                    print(f"e {i} |{iteration} it: {avg_loss.item() / 100}")
                    writer.add_scalar('training loss', avg_loss.item() / 100, iteration)
                    avg_loss = 0

                if iteration % 1000 == 1:
                    pred = decode_segmap(pred[0].cpu().data.numpy())
                    gt = decode_segmap(targets[0].cpu().data.numpy())

                    pred = torch.from_numpy(pred).permute(2, 0, 1)
                    gt = torch.from_numpy(gt).permute(2, 0, 1)
                    writer.add_image("pred", pred, iteration)
                    writer.add_image("gt", gt, iteration)

    def validation(self, it, e):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        torch.cuda.empty_cache()

        self.model.eval()
        _preds = []
        _targets = []
        print("Evaluating")
        for image, target, _ in tqdm(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
            self.metric.update(outputs, target)
            pixAcc, mIoU = self.metric.get()

            pred = torch.max(outputs, 1).indices
            for i in range(pred.shape[0]):
                if len(_preds) < 64:
                    _preds.append(torchvision.transforms.ToTensor()(decode_segmap(pred[i].cpu().data.numpy())))
                    _targets.append(torchvision.transforms.ToTensor()(decode_segmap(target[i].cpu().data.numpy())))

        _preds = torchvision.utils.make_grid(_preds, nrow=8)
        _targets = torchvision.utils.make_grid(_targets, nrow=8)

        new_pred = (pixAcc + mIoU) / 2
        print(f"pixel acc: {pixAcc}\nmIoU: {mIoU}")
        writer.add_scalar('validation pixAcc', pixAcc, it)
        writer.add_scalar('validation mIoU', mIoU, it)
        writer.add_image("gt", _targets, it)
        writer.add_image("pred", _preds, it)

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"dinov2_mscoco.pth"
    filename = os.path.join(directory, filename)

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'dinov2_mscoco_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    # reference maskrcnn-benchmark
    args.device = "cuda"
    writer = SummaryWriter()
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
