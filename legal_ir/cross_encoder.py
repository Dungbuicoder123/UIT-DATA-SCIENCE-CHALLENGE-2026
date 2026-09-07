"""
LegalIR — Stage 3: Cross-Encoder Re-ranker (MỚI)
"""
import logging
from typing import List, Tuple
logger = logging.getLogger(__name__)

class CrossEncoderRanker:
    def __init__(self, model_name: str, use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None

    def _load_model(self):
        if self._model is not None: return
        from sentence_transformers import CrossEncoder
        device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Loading Cross-Encoder model: {self.model_name}")
        self._model = CrossEncoder(self.model_name, device=device, max_length=2048)

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """
        candidates: List of (chunk_id, chunk_text)
        Trả về: [(chunk_id, score), ...] sắp xếp giảm dần.
        """
        if not candidates: return []
        self._load_model()
        pairs = [[query, text] for _, text in candidates]
        
        # Cross-encoder tính toán trực tiếp cặp (Query, Document)
        scores = self._model.predict(pairs)
        
        results = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
