from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting, AppSettings
from models.keyframe import Keyframe
from factory.factory import ServiceFactory
from core.logger import SimpleLogger

mongo_client: AsyncIOMotorClient = None
service_factory: ServiceFactory = None
logger = SimpleLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    logger.info("Starting up application...")
    
    try:
        # Initialize settings
        app_settings = AppSettings()
        milvus_settings = KeyFrameIndexMilvusSetting()
        
        # Initialize MongoDB settings
        try:
            settings = MongoDBSettings()
            logger.debug(f"MongoDB settings: username={settings.MONGO_USER}, db={settings.MONGO_DB}")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDBSettings: {e}")
            raise
        
        # MongoDB Atlas connection string
        mongo_uri = (
            f"mongodb+srv://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@"
            f"{settings.MONGO_HOST}/{settings.MONGO_DB}?retryWrites=true&w=majority"
        )
        global mongo_client
        mongo_client = AsyncIOMotorClient(
            mongo_uri,
            serverSelectionTimeoutMS=60000,
            connectTimeoutMS=60000,
            socketTimeoutMS=60000
        )
        try:
            # Test connection
            await mongo_client.server_info()
            logger.info("Successfully connected to MongoDB Atlas")
            await init_beanie(database=mongo_client[settings.MONGO_DB], document_models=[Keyframe])
            logger.info("Initialized Beanie with Keyframe model")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise
        
        global service_factory
        milvus_search_params = {
            "metric_type": milvus_settings.METRIC_TYPE,
            "params": milvus_settings.SEARCH_PARAMS
        }
        
        # # Test Milvus connection
        # try:
        #     from pymilvus import connections
        #     import ssl
        #     conn_params = {
        #         "host": milvus_settings.HOST,
        #         "port": int(milvus_settings.PORT),  # Ensure port is int
        #         "timeout": 10,  # 10-second timeout
        #         "secure": True,  # Enable secure connection
        #         "ssl": True,     # Explicitly enable SSL
        #         "ssl_version": ssl.TLSVersion.TLSv1_2,  # Enforce TLS 1.2
        #         "retry_times": 3  # Retry 3 times
        #     }
        #     for attempt in range(conn_params["retry_times"]):
        #         try:
        #             connections.connect(alias="default", **conn_params)
        #             logger.info(f"Successfully connected to Milvus at {milvus_settings.HOST}:{milvus_settings.PORT} on attempt {attempt + 1}")
        #             break
        #         except Exception as e:
        #             logger.error(f"Attempt {attempt + 1} failed to connect to Milvus: {e}")
        #             if attempt == conn_params["retry_times"] - 1:
        #                 raise
        #             import time
        #             time.sleep(2)  # Wait 2 seconds before retry
        #     # Verify collection existence
        #     from pymilvus import utility
        #     if not utility.has_collection(milvus_settings.COLLECTION_NAME, using="default"):
        #         logger.warning(f"Collection {milvus_settings.COLLECTION_NAME} not found in Milvus")
        # except Exception as e:
        #     logger.error(f"Failed to connect to Milvus: {e}")
        #     raise
        
        service_factory = ServiceFactory(
            milvus_collection_name=milvus_settings.COLLECTION_NAME,
            milvus_host=milvus_settings.HOST,
            milvus_port=milvus_settings.PORT,
            milvus_user="",  
            milvus_password="",  
            milvus_search_params=milvus_search_params,
            model_name=app_settings.MODEL_NAME,
            mongo_collection=Keyframe
        )
        logger.info("Service factory initialized successfully")
        
        app.state.service_factory = service_factory
        app.state.mongo_client = mongo_client
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    logger.info("Shutting down application...")
    
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed")
        from pymilvus import connections
        if connections.has_connection("default"):
            connections.disconnect("default")
            logger.info("Milvus connection closed")
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
