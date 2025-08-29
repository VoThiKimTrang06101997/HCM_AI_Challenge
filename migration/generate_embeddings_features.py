import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import json
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

# Paths
FEATURE_DIR = r"D:\AI Viet Nam\AI_Challenge\Dataset\clip-features-32"
ID2INDEX_PATH = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\id2index.json"
OUTPUT_PATH = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\CLIP_ViT-B-32_laion2b_s34b_b79k_clip_embeddings.pt"
KEYFRAME_DIR = r"D:\AI Viet Nam\AI_Challenge\Dataset\Keyframes"

def generate_embeddings():
    # Load id2index.json for keyframe paths
    try:
        with open(ID2INDEX_PATH, 'r') as f:
            id2index = json.load(f)
        valid_groups = list(range(21, 31))  # Groups L21 to L30
        id2index = {k: v for k, v in id2index.items() if int(v.split('/')[0]) in valid_groups}
        print(f"Loaded {len(id2index)} entries from id2index.json for groups L21 to L30")
    except Exception as e:
        print(f"Error loading id2index.json: {e}")
        return

    # Initialize model
    try:
        model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()  # Set model to evaluation mode
        tokenizer = get_tokenizer('ViT-B-32')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Initialized CLIP model 'laion2b_s34b_b79k' on {device}")
    except Exception as e:
        print(f"Error initializing CLIP model 'laion2b_s34b_b79k': {e}")
        print(f"Check if the model tag is correct or required dependencies are installed.")
        return

    # Initialize embedding list
    embedding_dim = 512  # Dimension for ViT-B-32
    embeddings = np.zeros((len(id2index), embedding_dim), dtype=np.float32)
    key_to_index = {k: i for i, k in enumerate(sorted(id2index.keys(), key=int))}  # Sort by key for consistency

    # Process each keyframe image based on id2index
    processed_keys = set()
    for key, value in tqdm(id2index.items(), desc="Processing keyframes"):
        try:
            g_num, v_num, k_num = map(int, value.split('/'))
            filename = f"{k_num:03d}"
            base_path = os.path.join(KEYFRAME_DIR, f"Keyframes_L{g_num:02d}", "keyframes", f"L{g_num:02d}_V{v_num:03d}")
            for ext in ('.jpg', '.png'):
                img_path = os.path.join(base_path, f"{filename}{ext}")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = model.encode_image(img_tensor).cpu().numpy()[0]  # (512,)
                    index = key_to_index[key]
                    if 0 <= index < len(embeddings):
                        embeddings[index] = embedding
                        processed_keys.add(key)
                    else:
                        print(f"Skipping key {key}: index {index} out of range for embeddings array")
                    break
            else:
                print(f"No image found for key {key} at {base_path}")
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            continue

    print(f"Processed {len(processed_keys)} out of {len(id2index)} keys")
    if len(processed_keys) != len(id2index):
        print(f"Warning: Not all id2index keys were matched. Missing {len(id2index) - len(processed_keys)} keys")
    
    # Convert to torch tensor and save
    try:
        embeddings_tensor = torch.from_numpy(embeddings)
        torch.save(embeddings_tensor, OUTPUT_PATH)
        print(f"Saved embeddings to {OUTPUT_PATH} with shape {embeddings_tensor.shape}")
    except Exception as e:
        print(f"Error saving embeddings to {OUTPUT_PATH}: {e}")
        return

if __name__ == "__main__":
    generate_embeddings()
