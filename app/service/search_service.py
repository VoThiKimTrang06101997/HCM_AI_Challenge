import os
import sys
import numpy as np

try:
    from core.logger import SimpleLogger, logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    SimpleLogger = logging.getLogger

logger = SimpleLogger(__name__)
# Add root directory to Python path
ROOT_FOLDER = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline"
sys.path.insert(0, ROOT_FOLDER)

# Debug directory structure
logger.debug(f"ROOT_FOLDER: {ROOT_FOLDER}")
logger.debug(f"sys.path: {sys.path}")
logger.debug(f"Directory contents of {ROOT_FOLDER}: {os.listdir(ROOT_FOLDER)}")
app_path = os.path.join(ROOT_FOLDER, 'app')
if os.path.exists(app_path):
    logger.debug(f"Contents of {app_path}: {os.listdir(app_path)}")
else:
    logger.error(f"app directory not found at {app_path}")

# Try importing Keyframe
try:
    from app.models.keyframe import Keyframe
    from app.core.settings import MongoDBSettings
except ImportError as e:
    logger.error(f"Failed to import Keyframe or MongoDBSettings: {e}")
    logger.info("Attempting alternative import paths...")
    try:
        from models.keyframe import Keyframe
        from core.settings import MongoDBSettings
        logger.info("Successfully imported Keyframe from models.keyframe")
    except ImportError as e:
        logger.error(f"Alternative import failed: {e}")
        raise


from repository.milvus import KeyframeVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from typing import List, Optional
from app.models.keyframe import Keyframe

from schema.response import KeyframeServiceResponse

class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo


    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])
  
        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]   
        return return_keyframe


    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None
    ) -> list[KeyframeServiceResponse]:
        
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes(sorted_ids)



        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_) 
            if keyframe is not None:
                response.append(
                    KeyframeServiceResponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance
                    )
                )
        return response
    

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)   
    

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int,int]]
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))
        
        
        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    


    




    
        



        

        

        
        
        


        

        







