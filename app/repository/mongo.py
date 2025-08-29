"""
The implementation of Keyframe repositories. The following class is responsible for getting the keyframe by many ways
"""

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from typing import Any
from models.keyframe import Keyframe
from common.repository import MongoBaseRepository
from schema.interface import KeyframeInterface

from pymongo import MongoClient
from app.core.settings import AppSettings


class KeyframeRepository(MongoBaseRepository[Keyframe]):
    # def __init__(self, settings: AppSettings):
    #     try:
    #         # Construct MongoDB connection string with SSL and timeouts
    #         connection_string = (
    #             f"mongodb+srv://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@"
    #             f"{settings.MONGO_HOST}/{settings.MONGO_DB}?retryWrites=true&w=majority"
    #             f"&tls=true&tlsAllowInvalidCertificates=false"
    #         )
    #         self.client = MongoClient(
    #             connection_string,
    #             serverSelectionTimeoutMS=30000,  # Increase to 30s
    #             connectTimeoutMS=30000,          # Increase to 30s
    #             socketTimeoutMS=30000            # Increase to 30s
    #         )
    #         self.db = self.client[settings.MONGO_DB]
    #         self.collection = self.db.keyframes
    #         print(f"Connected to MongoDB Atlas: {settings.MONGO_DB}")
    #     except Exception as e:
    #         print(f"Error connecting to MongoDB: {e}")
    #         raise
        
    async def get_keyframe_by_list_of_keys(
        self, keys: list[int]
    ):
        result = await self.find({"key": {"$in": keys}})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result

        ]

    async def get_keyframe_by_video_num(
        self, 
        video_num: int,
    ):
        result = await self.find({"video_num": video_num})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result
        ]

    async def get_keyframe_by_keyframe_num(
        self, 
        keyframe_num: int,
    ):
        result = await self.find({"keyframe_num": keyframe_num})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result
        ]   


