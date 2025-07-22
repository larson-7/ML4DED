gdown 12CXpzsbGS48UlS9r2mw6T_irihVh7RAa -O training_data.zip
unzip -o training_data.zip
python3 ml4ded/util/img_vid_utils/rle_mask_to_img.py \
    --data-dir training_data