# import torch
# import json
# embeddings = torch.load(r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\CLIP_ViT-B-32_laion2b_s34b_b79k_clip_embeddings.pt", map_location=torch.device('cpu'))
# with open(r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\id2index.json", 'r') as f:
#     id2index = json.load(f)
# print(len(embeddings), len(id2index))  # Should match

import os
from pathlib import Path

# Path to Keyframes directory
KEYFRAME_DIR = r"D:\AI Viet Nam\AI_Challenge\Dataset\Keyframes"

def count_images(keyframe_dir: str):
    keyframe_path = Path(keyframe_dir)
    if not keyframe_path.exists():
        print(f"Keyframes directory {keyframe_path} does not exist")
        return 0

    total_images = 0
    valid_groups = [f"Keyframes_L{i:02d}" for i in range(21, 31)]

    for batch_folder in valid_groups:
        batch_path = keyframe_path / batch_folder
        if not batch_path.exists():
            print(f"Batch folder {batch_path} does not exist")
            continue

        video_root = batch_path / 'keyframes'
        if not video_root.exists():
            print(f"Video root {video_root} does not exist")
            continue

        for video_folder in os.listdir(video_root):
            video_path = video_root / video_folder
            if video_path.is_dir():
                keyframe_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.png'))]
                count = len(keyframe_files)
                total_images += count
                print(f"{video_path}: {count} images")

    print(f"Total number of images (.jpg and .png) in {keyframe_dir}: {total_images}")
    return total_images

if __name__ == "__main__":
    total = count_images(KEYFRAME_DIR)
    if total != 289324:
        print(f"Warning: Total images ({total}) does not match expected count (289324). Dataset may be incomplete.")
        