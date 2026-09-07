"""
LegalIR — Stage 2: Dense Retriever
"""
import logging, os, pickle
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, model_name: str, use_gpu: bool, encode_batch_size: int, max_seq_length: int = 2048, embeddings_cache_path: Optional[str] = None, doc_ids_cache_path: Optional[str] = None):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.encode_batch_size = encode_batch_size
        self.max_seq_length = max_seq_length
        self.embeddings_cache_path = embeddings_cache_path
        self.doc_ids_cache_path = doc_ids_cache_path
        self._model = None
        self._embeddings = None
        self._chunk_ids = []
        self._id_to_idx = {}

    def _load_model(self):
        if self._model is not None: return
        from sentence_transformers import SentenceTransformer
        device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Loading dense model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=device)
        self._model.max_seq_length = self.max_seq_length # Khai thác tối đa 2048 tokens

    def encode_corpus(self, raw_texts: List[str], chunk_ids: List[str], force_rebuild: bool = False):
        self._chunk_ids = chunk_ids
        self._id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        
        if not force_rebuild and self.embeddings_cache_path and os.path.isfile(self.embeddings_cache_path) and self.doc_ids_cache_path and os.path.isfile(self.doc_ids_cache_path):
            self._embeddings = np.load(self.embeddings_cache_path)
            with open(self.doc_ids_cache_path, "rb") as f: cached_ids = pickle.load(f)
            if len(cached_ids) == len(chunk_ids):
                self._chunk_ids = cached_ids
                self._id_to_idx = {cid: i for i, cid in enumerate(cached_ids)}
                return

        self._load_model()
        self._embeddings = self._model.encode(raw_texts, batch_size=self.encode_batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        
        if self.embeddings_cache_path:
            os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
            np.save(self.embeddings_cache_path, self._embeddings)
        if self.doc_ids_cache_path:
            with open(self.doc_ids_cache_path, "wb") as f: pickle.dump(self._chunk_ids, f)

    def encode_queries_batch(self, query_texts: List[str]) -> np.ndarray:
        self._load_model()
        return self._model.encode(query_texts, batch_size=self.encode_batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def rerank(self, query_vec: np.ndarray, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        valid_ids = [cid for cid in candidate_ids if cid in self._id_to_idx]
        if not valid_ids: return []
        indices = np.array([self._id_to_idx[cid] for cid in valid_ids])
        scores = self._embeddings[indices] @ query_vec
        order = np.argsort(scores)[::-1]
        return [(valid_ids[i], float(scores[i])) for i in order]
