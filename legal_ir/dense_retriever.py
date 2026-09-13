"""
LegalIR — Stage 2: Dense Retriever
Nâng cấp: Thêm Full-Corpus ANN Search (không chỉ re-rank BM25 candidates)
"""
import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        use_gpu: bool,
        encode_batch_size: int,
        max_seq_length: int = 2048,
        embeddings_cache_path: Optional[str] = None,
        doc_ids_cache_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.encode_batch_size = encode_batch_size
        self.max_seq_length = max_seq_length
        self.embeddings_cache_path = embeddings_cache_path
        self.doc_ids_cache_path = doc_ids_cache_path
        self._model = None
        self._embeddings: Optional[np.ndarray] = None   # shape: (N, D), float32, L2-normalised
        self._chunk_ids: List[str] = []
        self._id_to_idx: dict = {}

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Loading dense model: {self.model_name} on {device}")
        self._model = SentenceTransformer(self.model_name, device=device)
        self._model.max_seq_length = self.max_seq_length

    # ------------------------------------------------------------------
    # Corpus encoding
    # ------------------------------------------------------------------

    def encode_corpus(
        self,
        raw_texts: List[str],
        chunk_ids: List[str],
        force_rebuild: bool = False,
    ):
        self._chunk_ids = chunk_ids
        self._id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

        # Try to load from cache
        if (
            not force_rebuild
            and self.embeddings_cache_path
            and os.path.isfile(self.embeddings_cache_path)
            and self.doc_ids_cache_path
            and os.path.isfile(self.doc_ids_cache_path)
        ):
            self._embeddings = np.load(self.embeddings_cache_path)
            with open(self.doc_ids_cache_path, "rb") as f:
                cached_ids = pickle.load(f)
            if len(cached_ids) == len(chunk_ids):
                logger.info(
                    f"Loaded dense embeddings from cache: {self._embeddings.shape}"
                )
                self._chunk_ids = cached_ids
                self._id_to_idx = {cid: i for i, cid in enumerate(cached_ids)}
                return

        # Encode from scratch
        self._load_model()
        logger.info(f"Encoding {len(raw_texts)} chunks with {self.model_name}...")
        self._embeddings = self._model.encode(
            raw_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,     # L2-normalise → cosine sim = dot product
            convert_to_numpy=True,
        ).astype(np.float32)

        # Save cache
        if self.embeddings_cache_path:
            os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
            np.save(self.embeddings_cache_path, self._embeddings)
        if self.doc_ids_cache_path:
            with open(self.doc_ids_cache_path, "wb") as f:
                pickle.dump(self._chunk_ids, f)

        logger.info(f"Corpus embeddings shape: {self._embeddings.shape}")

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def encode_queries_batch(self, query_texts: List[str]) -> np.ndarray:
        """Encode một batch câu hỏi → ma trận (Q, D) đã L2-normalise."""
        self._load_model()
        return self._model.encode(
            query_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # ★ NEW: Full-Corpus Search (True ANN bằng matrix dot-product)
    # ------------------------------------------------------------------

    def search_full_corpus(
        self, query_vec: np.ndarray, top_k: int = 150
    ) -> List[Tuple[str, float]]:
        """
        Tìm kiếm Top-K chunks có độ tương đồng cao nhất với query_vec
        trong TOÀN BỘ corpus — không phụ thuộc vào BM25.

        Vì embeddings đã L2-normalised, dot-product = cosine similarity.
        Dùng numpy argpartition để tránh sort toàn bộ N phần tử.

        Args:
            query_vec: vector truy vấn đã normalise, shape (D,)
            top_k:     số chunk muốn lấy

        Returns:
            List[(chunk_id, score)] sắp xếp giảm dần theo score.
        """
        if self._embeddings is None:
            return []

        # (N, D) @ (D,) → (N,)
        scores: np.ndarray = self._embeddings @ query_vec
        n = len(scores)
        k = min(top_k, n)

        # argpartition: O(N) thay vì O(N log N)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Legacy: Re-rank a given candidate list (vẫn giữ để tương thích)
    # ------------------------------------------------------------------

    def rerank(
        self, query_vec: np.ndarray, candidate_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Tính lại điểm cho một tập candidate chunk_ids cụ thể.
        Dùng trong trường hợp muốn score lại danh sách từ BM25.
        """
        valid_ids = [cid for cid in candidate_ids if cid in self._id_to_idx]
        if not valid_ids:
            return []
        indices = np.array([self._id_to_idx[cid] for cid in valid_ids])
        scores = self._embeddings[indices] @ query_vec
        order = np.argsort(scores)[::-1]
        return [(valid_ids[i], float(scores[i])) for i in order]
