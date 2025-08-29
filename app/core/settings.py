from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()


class MongoDBSettings(BaseSettings):
    MONGO_HOST: str = "cluster0.gml39fb.mongodb.net"
    MONGO_PORT: int = 27017
    MONGO_DB: str = "HCM_AI_Challenge"
    MONGO_USER: str = "vothikimtrang"
    MONGO_PASSWORD: str = "06101997"
    
    def get_mongo_uri(self) -> str:
        """Generate MongoDB Atlas connection string."""
        return f"mongodb+srv://{self.MONGO_USER}:{self.MONGO_PASSWORD}@{self.MONGO_HOST}/{self.MONGO_DB}?retryWrites=true&w=majority"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class IndexPathSettings(BaseSettings):
    FAISS_INDEX_PATH: str | None  
    USEARCH_INDEX_PATH: str | None

class KeyFrameIndexMilvusSetting(BaseSettings):
    COLLECTION_NAME: str = "keyframe"
    HOST: str = 'localhost'
    PORT: str = '19530'
    METRIC_TYPE: str = 'COSINE'
    INDEX_TYPE: str = 'FLAT'
    BATCH_SIZE: int =10000
    SEARCH_PARAMS: dict = {}
    
class AppSettings(BaseSettings):
    # ASR_PATH: str = '/media/tinhanhnguyen/Data3/Projects/HCMAI2025_Baseline/app/data/asr_proc.json'
    
    # DATA_FOLDER: str = Field(
    #     "D:\\AI Viet Nam\\AI_Challenge\\Dataset\\Keyframes",
    #     description="Root directory for keyframe data, structured as Keyframes_Lxx\keyframes\Lxx_Vyyy\zzz.jpg"
    # )
    
    ROOT_FOLDER: str = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline"
    DATA_FOLDER: str = r"D:\AI Viet Nam\AI_Challenge\Dataset\Keyframes"
    ID2INDEX_PATH: str = r"D:\AI Viet Nam\AI_Challenge\Source_Code\HCMAI2025_Baseline\file_embeddings\id2index.json"
    CLIP_FEATURES_PATH: str = r"D:\AI Viet Nam\AI_Challenge\Dataset\clip-features-32"
    FRAME2OBJECT: str = r"D:\AI Viet Nam\AI_Challenge\Dataset\objects"
    MODEL_NAME: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    

    def __init__(self, **values):
        super().__init__(**values)
        # Handle tuple input for ROOT_FOLDER
        if isinstance(self.ROOT_FOLDER, tuple):
            self.ROOT_FOLDER = self.ROOT_FOLDER[0] if self.ROOT_FOLDER else self.ROOT_FOLDER
            if not isinstance(self.ROOT_FOLDER, str):
                raise ValueError(f"ROOT_FOLDER must be a string, got {type(self.ROOT_FOLDER)}: {self.ROOT_FOLDER}")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


