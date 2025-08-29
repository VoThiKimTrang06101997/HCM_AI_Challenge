from typing import Tuple, Union, List
from schema.response import KeyframeServiceResponse
from service import ModelService, KeyframeQueryService
from pathlib import Path
import json
import numpy as np
import os
import sys
import csv

# Add root directory to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

try:
    from core.logger import SimpleLogger, logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    SimpleLogger = logging.getLogger

logger = SimpleLogger(__name__)


class QueryController:
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.output_dir = Path(r"D:\AI Viet Nam\AI_Challenge\Result")
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_model_to_path(
        self,
        model: KeyframeServiceResponse
    ) -> tuple[str, float]:
        base_path = os.path.join(
            self.data_folder, f"Keyframes_L{model.group_num:02d}", "keyframes", f"L{model.group_num:02d}_V{model.video_num:03d}")
        filename = f"{model.keyframe_num:03d}"
        for ext in ('.jpg', '.png'):
            path = os.path.join(base_path, f"{filename}{ext}")
            if os.path.exists(path):
                return path, model.confidence_score
        return os.path.join("static", "path_not_found.jpg"), model.confidence_score

    def _format_to_csv_row(self, item: Union[KeyframeServiceResponse, Tuple]) -> Tuple[str, int]:
        if isinstance(item, KeyframeServiceResponse):
            path, score = self.convert_model_to_path(item)
            video_id = f"L{item.group_num:02d}_V{item.video_num:03d}"
            frame_idx = item.keyframe_num
            return video_id, frame_idx
        elif isinstance(item, tuple) and len(item) == 2:
            path, score = item
            # Derive video_id and frame_idx from path if possible
            for key, value in self.id2index.items():
                if str(path).find(f"L{value.split('/')[0]:02d}_V{value.split('/')[1]:03d}") != -1:
                    group_num, video_num, frame_idx = map(
                        int, value.split('/'))
                    video_id = f"L{group_num:02d}_V{video_num:03d}"
                    return video_id, frame_idx
        return "Unknown", 0

    async def search_text(
        self,
        query: str,
        top_k: int,
        score_threshold: float
    ):
        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)

        # Convert to CSV with limit of 100
        output_file = self.output_dir / \
            f"search_text_{query.replace(' ', '_')[:75]}_{top_k}_{score_threshold}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_id", "frame_idx"])  # Header
            for idx, item in enumerate(result[:100]):  # Limit to 100
                video_id, frame_idx = self._format_to_csv_row(item)
                writer.writerow([video_id, frame_idx])
        logger.info(f"Saved {min(100, len(result))} results to {output_file}")
        return result

    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int]
    ):
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        embedding = self.model_service.embedding(query).tolist()[0]

        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)

        # Convert to CSV with limit of 100
        output_file = self.output_dir / \
            f"search_text_exclude_{query.replace(' ', '_')[:75]}_{top_k}_{score_threshold}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_id", "frame_idx"])
            for idx, item in enumerate(result[:100]):  # Limit to 100
                video_id, frame_idx = self._format_to_csv_row(item)
                writer.writerow([video_id, frame_idx])
        logger.info(f"Saved {min(100, len(result))} results to {output_file}")
        return result

    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int]
    ):

        exclude_ids = None
        if len(list_of_include_groups) > 0 and len(list_of_include_videos) == 0:
            print("hi")
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) > 0:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) == 0:
            exclude_ids = []
        else:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if (
                    int(v.split('/')[0]) not in list_of_include_groups or
                    int(v.split('/')[1]) not in list_of_include_videos
                )
            ]

        embedding = self.model_service.embedding(query).tolist()[0]

        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)

        # Convert to CSV with limit of 100
        output_file = self.output_dir / \
            f"search_selected_{query.replace(' ', '_')[:75]}_{top_k}_{score_threshold}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_id", "frame_idx"])
            for idx, item in enumerate(result[:100]):  # Limit to 100
                video_id, frame_idx = self._format_to_csv_row(item)
                writer.writerow([video_id, frame_idx])
        logger.info(f"Saved {min(100, len(result))} results to {output_file}")
        return result
