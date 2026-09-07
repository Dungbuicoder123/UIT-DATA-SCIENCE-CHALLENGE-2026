"""
LegalIR — Stage 1: BM25
"""
import logging, os, pickle
from typing import List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from tqdm import tqdm

logger = logging.getLogger(__name__)
_BM25_VARIANTS = {"BM25Okapi": BM25Okapi, "BM25L": BM25L, "BM25Plus": BM25Plus}

class BM25Retriever:
    def __init__(self, variant: str = "BM25Okapi", cache_path: Optional[str] = None):
        self.variant, self.cache_path = variant, cache_path
        self._bm25, self._chunk_ids = None, []

    def build_index(self, bm25_texts: List[str], chunk_ids: List[str], force_rebuild: bool = False):
        self._chunk_ids = chunk_ids
        if not force_rebuild and self.cache_path and os.path.isfile(self.cache_path):
            self._load_cache()
            if len(self._chunk_ids) == len(chunk_ids): return
        
        tokenized = [text.split() for text in tqdm(bm25_texts, desc="Tokenizing cho BM25")]
        self._bm25 = _BM25_VARIANTS[self.variant](tokenized)
        if self.cache_path: self._save_cache()

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        query_tokens = query.split()
        scores = self._bm25.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Bỏ Normalize vì RRF tính theo rank, không cần score min-max nữa
        top_scores = scores[top_indices]
        return [(self._chunk_ids[idx], float(top_scores[i])) for i, idx in enumerate(top_indices)]

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f: pickle.dump({"bm25": self._bm25, "cids": self._chunk_ids}, f)

    def _load_cache(self):
        with open(self.cache_path, "rb") as f: cached = pickle.load(f)
        self._bm25, self._chunk_ids = cached["bm25"], cached["cids"]
