import open_clip
import torch
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


class ModelService:
    def __init__(
        self,
        model,
        preprocess,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.model = model.to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def embedding(self, query_text: str) -> np.ndarray:
        """
        Return (1, ndim 1024) torch.Tensor
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([query_text]).to(self.device)
            query_embedding = self.model.encode_text(
                # (1, 512)
                text_tokens).cpu().detach().numpy().astype(np.float32)
        return query_embedding
