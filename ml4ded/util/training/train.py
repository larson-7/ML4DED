import argparse
import os
import shutil
import sys
from enum import Enum
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from ml4ded.models.dino2seg import Dino2Seg
from ml4ded.util.training.segmentationMetric import *
from ml4ded.util.vis import decode_segmap
from ml4ded.util.dataset.ml4ded_seg_dataset import ML4DEDSegmentationDataset
from ml4ded.util.training.early_stopping import EarlyStopping
from ml4ded.util.dataset.augmentations.augmentations import get_train_augmentation, get_val_augmentation


class SegLabels(Enum):
    BACKGROUND = 0
    HEAD = 1
    BASEPLATE = 2
    PREVIOUS_PART = 3
    CURRENT_PART = 4
    WELD_FLASH = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--data-dir', type=str, default="./data/ml4ded",
                        help='train/test data directory')
    parser.add_argument('--model-weights-dir', type=str, default="./model_weights",
                        help='pretrained model weights directory')

    parser.add_argument('--base-size', type=int, default=580,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=518,
                        help='crop image size')

    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--save-dir', default='./ckpt', help='Directory for saving checkpoint models')
    parser.add_argument('--device', default='cuda', help='Training device')
    return parser.parse_args()


def make_divisible(val, divisor=14):
    return val - (val % divisor)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.early_stopper = EarlyStopping(patience=30, delta=0.01, verbose=True)

        dataset_class = ML4DEDSegmentationDataset
        default_data_dir = os.path.join(root_path, "data/ml4ded")
        # Adapt to your actual image size
        img_h, img_w = make_divisible(1072), make_divisible(608)

        data_dir = args.data_dir if args.data_dir else default_data_dir

        # image transform (normalize to imagenet mean statistics)
        train_transform = get_train_augmentation(img_h, img_w)
        val_transform = get_val_augmentation(img_h, img_w)

        temporal_window = 4
        # dataset and dataloader
        trainset = dataset_class(data_dir, split="train", mode="train", temporal_window=temporal_window, transform=train_transform)
        valset = dataset_class(data_dir, split="test", mode="val",temporal_window=temporal_window,  transform=val_transform)
        
        self.train_loader = data.DataLoader(dataset=trainset, batch_size=args.batch_size, pin_memory=True)
        self.val_loader = data.DataLoader(dataset=valset, batch_size=args.batch_size, pin_memory=True)

        self.model = Dino2Seg(
            encoder="vitb",
            num_classes=len(trainset.classes),
            image_height=img_h,
            image_width=img_w,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            model_weights_dir=args.model_weights_dir,
            use_clstoken=True,
            use_temporal_consistency=True,
            num_temporal_tokens=16,
            temporal_window=temporal_window,
            cross_attn_heads=4,
            device=self.device,
        )

        class_weights = []
        for seg_label in SegLabels:
            match seg_label.name:
                case "BACKGROUND":
                    class_weights.append(0.1)
                case "HEAD":
                    class_weights.append(0.1)
                case "BASEPLATE":
                    class_weights.append(0.1)
                case "PREVIOUS_PART":
                    class_weights.append(0.1)
                case "CURRENT_PART":
                    class_weights.append(0.5)
                case "WELD_FLASH":
                    class_weights.append(0.1)

        class_weights = torch.FloatTensor(class_weights).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.metric = SegmentationMetric(len(trainset.classes), class_weights)
        self.best_pred = -1

    def setup_training_schedule(self, epoch):
        """Setup training schedule based on current epoch"""
        if epoch < 10:  # First 10 epochs: only train temporal components
            print(f"Epoch {epoch + 1}: Training only temporal components")

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only temporal-related parameters
            temporal_params = []

            # Unfreeze temporal tokens if they exist
            if hasattr(self.model.seg_head, 'temporal_extractor'):
                for param in self.model.seg_head.temporal_extractor.parameters():
                    param.requires_grad = True
                    temporal_params.append(param)

            if hasattr(self.model.seg_head, 'gate'):
                param = self.model.seg_head.gate
                param.requires_grad = True
                temporal_params.append(param)

            # Unfreeze cross-attention layers for temporal processing
            if hasattr(self.model.seg_head, 'cross_attn_block'):
                for param in self.model.seg_head.cross_attn_block.parameters():
                    param.requires_grad = True
                    temporal_params.append(param)

            # Create new optimizer with only temporal parameters
            if temporal_params:
                self.optimizer = torch.optim.AdamW(temporal_params, lr=self.args.lr)
                print(f"Training {len(temporal_params)} temporal parameters")
            else:
                print("Warning: No temporal parameters found!")

        else:  # After epoch 10: train everything
            print(f"Epoch {epoch + 1}: Training all parameters")

            # Unfreeze all parameters (except encoder which stays frozen)
            for name, param in self.model.named_parameters():
                if 'pretrained' not in name.lower():  # Keep encoder frozen
                    param.requires_grad = True

            # Create optimizer with all trainable parameters with a reduced learning rate
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr/10)
            print(f"Training {len(trainable_params)} total parameters")


    def train(self):
        iteration = 0
        avg_loss = 0
        for i in range(args.epochs):
            print("-------------------------------------------------------")
            print("Training Epoch {}/{}".format(i + 1, args.epochs))

            # Setup training schedule for this epoch
            self.setup_training_schedule(i)

            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for images, targets, _ in tqdm(self.train_loader):
                iteration += 1
                num_batches += 1

                # images: (B, T, 3, H, W), targets: (B, T, H, W)
                images = images.to(self.device)
                targets = targets.to(self.device)

                current_images = images[:, -1]  # (B, 3, H, W)
                current_targets = targets[:, -1]  # (B, H, W)

                prev_temporal_images = images[:, :-1]  # (B, T-1, 3, H, W)
                prev_temporal_images = prev_temporal_images.permute(1, 0, 2, 3, 4)  # (T-1, B, 3, H, W)

                previous_temporal_tokens = self.model.get_previous_temporal_tokens(prev_temporal_images)

                outputs, temporal_tokens , attn_weights= self.model(current_images, previous_temporal_tokens)  # shape (B, C, H, W)
                pred = torch.max(outputs, 1).indices

                loss = self.criterion(outputs, current_targets)
                loss = torch.mean(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss
                epoch_loss += loss.item()

                if iteration % 100 == 0:
                    patch_h, patch_w = current_images.shape[-2] // 14, current_images.shape[-1] // 14
                    print(f"epoch {i + 1} | iteration {iteration}: loss = {avg_loss.item() / 100:.4f}")
                    writer.add_scalar('training loss', avg_loss.item() / 100, iteration)
                    writer.add_scalar('temporal_gate', self.model.seg_head.gate.item(), iteration)

                    # ---------- TEMPORAL TOKENS ----------
                    temporal_tokens_vis = temporal_tokens[0].detach().cpu()  # (C, N_temp)
                    writer.add_image("temporal_tokens/heatmap", temporal_tokens_vis.unsqueeze(0), iteration)
                    writer.add_histogram("temporal_tokens/hist", temporal_tokens.detach().cpu(), iteration)

                    # ---------- TEMPORAL ATTENTION (Single Head) ----------
                    # attn_weights: (B, N_query, N_key)
                    attn_weights_b0 = attn_weights[0].detach().cpu()  # (N_query, N_key)

                    N_cls = 1 if self.model.use_clstoken else 0
                    N_spatial = patch_h * patch_w
                    temporal_start = N_cls + N_spatial

                    # Extract temporal token rows only
                    temporal_attn = attn_weights_b0[temporal_start:, :]  # (N_temp, N_key)

                    # Normalize for visualization
                    temporal_attn_norm = (temporal_attn - temporal_attn.min()) / (
                                temporal_attn.max() - temporal_attn.min() + 1e-6)
                    temporal_attn_img = temporal_attn_norm.unsqueeze(0)  # (1, 1, N_temp, N_key)
                    temporal_attn = temporal_attn.unsqueeze(0)

                    writer.add_image("attention_weights/temporal_only", temporal_attn_img, iteration)
                    writer.add_histogram("attention_weights/temporal_only_hist", temporal_attn, iteration)

                    avg_loss = 0  # Reset loss accumulator
                if iteration % 500 == 1:
                    pred_img = decode_segmap(pred[0].cpu().data.numpy())
                    gt_img = decode_segmap(targets[0, -1].cpu().data.numpy())  # Use current target
                    pred_img = torch.from_numpy(pred_img).permute(2, 0, 1)
                    gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)
                    writer.add_image("pred", pred_img, iteration)
                    writer.add_image("gt", gt_img, iteration)

            # Log epoch statistics and final gate values for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {i + 1} average loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('epoch_loss', avg_epoch_loss, i)

            # Validation
            val_metric = self.validation(iteration, i)

            self.early_stopper(val_metric)
            if self.early_stopper.early_stop:
                print(f"Early stopping at epoch {i + 1}")
                break



    def validation(self, it, e):
        is_best = False
        torch.cuda.empty_cache()
        self.model.eval()
        _preds, _targets = [], []

        print("Evaluating")
        self.metric.reset()

        for images, targets, _ in tqdm(self.val_loader):
            images = images.to(self.device)  # (B, T, 3, H, W)
            targets = targets.to(self.device)  # (B, T, H, W)

            current_images = images[:, -1]  # (B, 3, H, W)
            current_targets = targets[:, -1]  # (B, H, W)

            prev_temporal_images = images[:, :-1]  # (B, T-1, 3, H, W)
            prev_temporal_images = prev_temporal_images.permute(1, 0, 2, 3, 4)  # (T-1, B, 3, H, W)

            with torch.no_grad():
                previous_temporal_tokens = self.model.get_previous_temporal_tokens(prev_temporal_images)
                outputs, pred_tokens, attn_weights = self.model(current_images, previous_temporal_tokens)  # (B, C, H, W)
                preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            self.metric.update(outputs, current_targets)
            pixAcc, mIoU, weighted_mIoU = self.metric.get()

            for i in range(preds.shape[0]):
                if len(_preds) < 64:  # only log first few batches
                    _preds.append(torchvision.transforms.ToTensor()(decode_segmap(preds[i].cpu().numpy())))
                    _targets.append(torchvision.transforms.ToTensor()(decode_segmap(current_targets[i].cpu().numpy())))

        _preds = torchvision.utils.make_grid(_preds, nrow=8)
        _targets = torchvision.utils.make_grid(_targets, nrow=8)

        new_pred = (pixAcc + mIoU) / 2
        print(f"pixel acc: {pixAcc:.4f}\nmIoU: {mIoU:.4f}\nweighted mIoU: {weighted_mIoU:.4f}")
        writer.add_scalar('validation pixAcc', pixAcc, it)
        writer.add_scalar('validation mIoU', mIoU, it)
        writer.add_scalar('validation weighted mIoU', weighted_mIoU, it)
        # writer.add_image("val_gt", _targets, it)
        # writer.add_image("val_pred", _preds, it)

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        save_checkpoint(self.model, self.args, is_best)
        return new_pred


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"dinov2_seg.pth"
    filename = os.path.join(directory, filename)
    torch.save(model.seg_head.state_dict(), filename)
    if is_best:
        best_filename = 'dinov2_seg_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.device = "cuda"
    writer = SummaryWriter()
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
