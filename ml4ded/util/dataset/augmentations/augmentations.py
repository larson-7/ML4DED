import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentation(img_h, img_w, use_replay=True):
    base_transforms = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
        A.RandomCrop(height=img_h, width=img_w, p=0.3),
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    if use_replay:
        return A.ReplayCompose(base_transforms)
    else:
        return A.Compose(base_transforms)

def get_val_augmentation(img_h, img_w):
    return A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
