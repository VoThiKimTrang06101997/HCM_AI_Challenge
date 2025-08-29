import os
import json
import sys
import io
from pathlib import Path

# Add root directory to Python path
ROOT_DIR = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline-main"
sys.path.insert(0, ROOT_DIR)

try:
    from core.logger import SimpleLogger, logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    SimpleLogger = logging.getLogger

logger = SimpleLogger(__name__)

def generate_id2index(keyframes_root: str, output_path: str, expected_count: int = 289324):
    """
    Generate id2index.json from Keyframes directory with format like {"0": "24/1/137", "1": "24/1/138", ...}.
    
    Args:
        keyframes_root (str): Path to Keyframes directory (e.g., D:\AI Viet Nam\AI_Challenge\Dataset\Keyframes)
        output_path (str): Path to save id2index.json
        expected_count (int): Expected number of keyframes (default: 289324)
    """
    if os.name == 'nt':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    keyframes_root = Path(keyframes_root)
    if not keyframes_root.exists():
        logger.error(f"Keyframes directory {keyframes_root} does not exist")
        raise FileNotFoundError(f"Keyframes directory {keyframes_root} does not exist")

    id2index = {}
    global_id = 0

    try:
        # Explicitly process Keyframes_L21 to Keyframes_L30
        batch_folders = [f"Keyframes_L{i:02d}" for i in range(21, 31)]
        logger.info(f"Processing {len(batch_folders)} batch folders: {batch_folders}")
    except Exception as e:
        logger.error(f"Failed to define batch folders: {e}")
        raise

    for batch_folder in batch_folders:
        batch_path = keyframes_root / batch_folder
        if not batch_path.exists():
            logger.warning(f"Batch folder {batch_path} does not exist, skipping...")
            continue

        try:
            batch_num = int(batch_folder.split('_L')[1])  # L24 -> 24
        except (IndexError, ValueError) as e:
            logger.warning(f"Skipping invalid batch folder {batch_folder}: {e}")
            continue

        video_root = batch_path / 'keyframes'
        if not video_root.exists():
            logger.warning(f"Video root {video_root} does not exist, skipping...")
            continue

        try:
            video_folders = sorted(
                [f for f in os.listdir(video_root) if f.startswith("L") and '_V' in f],
                key=lambda x: int(x.split('_V')[1])
            )
            logger.debug(f"Found {len(video_folders)} video folders in {batch_folder}: {video_folders[:5]}...")
        except Exception as e:
            logger.warning(f"Failed to list video folders in {video_root}: {e}")
            continue

        for video_folder in video_folders:
            try:
                video_num = int(video_folder.split('_V')[1])
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping invalid video folder {video_folder}: {e}")
                continue

            video_path = video_root / video_folder
            if not video_path.exists():
                logger.warning(f"Video path {video_path} does not exist, skipping...")
                continue

            try:
                keyframe_files = sorted(
                    [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.png'))],
                    key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])))
                )
                logger.debug(f"Found {len(keyframe_files)} keyframe files in {video_folder}: {keyframe_files[:5]}...")
            except Exception as e:
                logger.warning(f"Failed to list keyframe files in {video_path}: {e}")
                continue

            for keyframe_file in keyframe_files:
                try:
                    # Extract keyframe index from filename, handling varied formats
                    keyframe_idx = int(''.join(filter(str.isdigit, keyframe_file.split('.')[0])))
                    value = f"{batch_num}/{video_num}/{keyframe_idx}"
                    id2index[str(global_id)] = value
                    global_id += 1
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping invalid keyframe file {keyframe_file}: {e}")
                    continue

    logger.info(f"Generated {global_id} id2index entries")
    if global_id != expected_count:
        logger.warning(f"Generated {global_id} entries, expected {expected_count}. Dataset may be incomplete or file parsing issue.")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(id2index, f, indent=4)
        logger.info(f"Saved id2index.json with {global_id} entries at {output_path}")
    except Exception as e:
        logger.error(f"Failed to save id2index.json to {output_path}: {e}")
        raise

    return id2index

if __name__ == "__main__":
    keyframes_root = r"D:\AI Viet Nam\AI_Challenge\Dataset\Keyframes"
    output_path = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\id2index.json"
    generate_id2index(keyframes_root, output_path)
