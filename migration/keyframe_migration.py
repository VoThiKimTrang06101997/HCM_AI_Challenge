import argparse
import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import sys
import os


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


SETTING = MongoDBSettings()


async def init_db():
    # Initialize MongoDB settings
    try:
        settings = MongoDBSettings()
        logger.debug(f"MongoDB settings: username={settings.MONGO_USER}, db={settings.MONGO_DB}")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDBSettings: {e}")
        raise
    
    # MongoDB Atlas connection string
    mongo_uri = f"mongodb+srv://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@cluster0.gml39fb.mongodb.net/{settings.MONGO_DB}?retryWrites=true&w=majority"
    client = AsyncIOMotorClient(mongo_uri)
    try:
        # Test connection
        await client.server_info()
        logger.info("Successfully connected to MongoDB Atlas")
        await init_beanie(database=client[settings.MONGO_DB], document_models=[Keyframe])
        logger.info("Initialized Beanie with Keyframe model")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        raise


def load_json_data(file_path):
    return json.load(open(file_path, 'r', encoding='utf-8'))


def transform_data(data: dict[str, str]) -> list[Keyframe]:
    """
    Convert the data from the old format to the new Keyframe model.
    """
    keyframes = []
    for key, value in data.items():
        group, video, keyframe = value.split('/')
        keyframe_obj = Keyframe(
            key=int(key),
            video_num=int(video),
            group_num=int(group),
            keyframe_num=int(keyframe)
        )
        keyframes.append(keyframe_obj)
    return keyframes


async def migrate_keyframes(file_path):
    await init_db()
    data = load_json_data(file_path)
    keyframes = transform_data(data)

    await Keyframe.delete_all()

    await Keyframe.insert_many(keyframes)
    print(f"Inserted {len(keyframes)} keyframes into the database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate keyframes to MongoDB.")
    parser.add_argument(
        "--file_path", type=str, help="Path to the JSON file containing keyframe data."
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File {args.file_path} does not exist.")
        sys.exit(1)

    asyncio.run(migrate_keyframes(args.file_path))
