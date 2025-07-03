import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from depth_anything_v2.seg_deformable_depth import SegmentationDeformableDepth
from util.vis import decode_segmap  # Your utility to convert segmentation to RGB

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def load_image(image_path, size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    return np.array(img)

def load_label(label_path, size):
    label = Image.open(label_path).resize(size, resample=Image.NEAREST)
    return np.array(label)

def run_model(model, image_tensor, device):
    with torch.no_grad():
        _, seg_map = model.infer_image(image_tensor)
    return seg_map[0]

def prepare_tensor(image_pil, img_w, img_h):
    transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return transform(image_pil).unsqueeze(0)

def build_figure(image_names, rgb_dir, seg_dir, model_weights_dir, output_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Preload one image to infer size
    raw_img = Image.open(os.path.join(rgb_dir, image_names[0])).convert("RGB")
    img_w, img_h = make_divisible(raw_img.size)

    # Load model
    model = SegmentationDeformableDepth(
        encoder="vitb",
        num_classes=6,
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=model_weights_dir,
        device=device,
    )
    model.eval().to(device)

    n = len(image_names)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7))

    for idx, img_name in enumerate(image_names):
        rgb_path = os.path.join(rgb_dir, img_name)
        label_path = os.path.join(seg_dir, img_name.replace("rgb_", "seg_"))

        # --- Load Image & Label ---
        raw_img = Image.open(rgb_path).convert("RGB")
        img_w, img_h = make_divisible(raw_img.size)
        img_np = np.array(raw_img.resize((img_w, img_h)))

        label_np = load_label(label_path, (img_w, img_h))
        label_rgb = decode_segmap(label_np, nc=8)

        input_tensor = prepare_tensor(raw_img, img_w, img_h).to(device)
        pred_np = run_model(model, input_tensor, device)
        pred_rgb = decode_segmap(pred_np, nc=8)

        # --- Plot Each Column ---
        for row, content, title in zip(
            [0, 1, 2],
            [img_np, label_rgb, pred_rgb],
            ["Image", "Label", "PiCIE (Baseline)"]
        ):
            ax = axes[row, idx] if n > 1 else axes[row]
            ax.imshow(content)
            ax.set_title(title if idx == 0 else "")
            ax.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-names', type=str, nargs='+', required=True, help='List of rgb image filenames')
    parser.add_argument('--rgb-dir', type=str, required=True, help='Directory with rgb images')
    parser.add_argument('--seg-dir', type=str, required=True, help='Directory with segmentation masks')
    parser.add_argument('--model-weights-dir', type=str, required=True, help='Pretrained model weights dir')
    parser.add_argument('--output-path', type=str, default=None, help='Optional path to save figure')
    args = parser.parse_args()

    build_figure(
        args.image_names,
        args.rgb_dir,
        args.seg_dir,
        args.model_weights_dir,
        output_path=args.output_path
    )
