import os

def get_model_weights(model_weights_dir, use_temporal, repo_id="iknocodes/ml4ded"):
    from huggingface_hub import hf_hub_download
    vitb_model_name = "dinov2_vitb14_reg4_pretrain.pth"
    os.makedirs(model_weights_dir, exist_ok=True)
    # Check for backbone
    vitb_weight_file = os.path.join(model_weights_dir, vitb_model_name)
    if not os.path.exists(vitb_weight_file):
        print("Downloading ViT-b backbone weights from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename=vitb_model_name, local_dir=model_weights_dir, local_dir_use_symlinks=False)
    # Seg head
    if use_temporal:
        seg_file = "ml4ded_seg_temporal.pth"
    else:
        seg_file = "ml4ded_seg.pth"
    seg_weight_file = os.path.join(model_weights_dir, seg_file)
    if not os.path.exists(seg_weight_file):
        print(f"Downloading {seg_file} from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename=seg_file, local_dir=model_weights_dir, local_dir_use_symlinks=False)
