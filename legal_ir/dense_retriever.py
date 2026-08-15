"""
=============================================================================
LegalIR — Stage 2: Dense Retriever (Sentence-Transformer)
=============================================================================
• Encode toàn bộ corpus thành dense vectors (embeddings)
• Hỗ trợ cache embeddings ra đĩa (numpy .npy)
• Tính cosine similarity giữa query và corpus candidates
• Hoạt động theo chế độ Re-ranker: chỉ tính similarity với Top-K từ BM25
=============================================================================
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retriever dựa trên Sentence-Transformer, hỗ trợ:
      - Encode corpus 1 lần, cache ra file .npy
      - Re-rank danh sách candidate (từ BM25) theo cosine similarity
      - GPU acceleration nếu có
    """

    def __init__(
        self,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        use_gpu: bool = True,
        encode_batch_size: int = 64,
        embeddings_cache_path: Optional[str] = None,
        doc_ids_cache_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name            : HuggingFace model ID hoặc đường dẫn local
            use_gpu               : Dùng GPU nếu có (CUDA)
            encode_batch_size     : Batch size khi encode corpus
            embeddings_cache_path : Path lưu/đọc embeddings matrix (.npy)
            doc_ids_cache_path    : Path lưu/đọc danh sách doc_ids (.pkl)
        """
        self.model_name            = model_name
        self.use_gpu               = use_gpu
        self.encode_batch_size     = encode_batch_size
        self.embeddings_cache_path = embeddings_cache_path
        self.doc_ids_cache_path    = doc_ids_cache_path

        self._model     = None              # SentenceTransformer model
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self._doc_ids: List[int] = []
        # Mapping: doc_id → row index trong _embeddings
        self._id_to_idx: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Model loader (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load Sentence-Transformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError(
                "Cài đặt sentence-transformers:\n"
                "  pip install sentence-transformers"
            )

        device = "cpu"
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("Dùng GPU (CUDA) để encode.")
                else:
                    logger.info("CUDA không khả dụng, dùng CPU.")
            except ImportError:
                logger.info("PyTorch chưa cài đặt, dùng CPU.")

        logger.info(f"Loading dense model: {self.model_name} ...")
        self._model = SentenceTransformer(self.model_name, device=device)
        logger.info("Dense model loaded.")

    # ------------------------------------------------------------------
    # Encode corpus
    # ------------------------------------------------------------------

    def encode_corpus(
        self,
        raw_texts: List[str],
        doc_ids: List[int],
        force_rebuild: bool = False,
    ) -> None:
        """
        Encode toàn bộ corpus thành dense embeddings.

        Args:
            raw_texts   : Văn bản gốc (đã clean, chưa tách từ —
                          Sentence-Transformer tự tokenize)
            doc_ids     : Danh sách doc_id tương ứng
            force_rebuild: Bỏ qua cache, encode lại từ đầu
        """
        if len(raw_texts) != len(doc_ids):
            raise ValueError("raw_texts và doc_ids phải có cùng độ dài.")

        self._doc_ids  = doc_ids
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}

        # Thử load từ cache
        if not force_rebuild \
                and self.embeddings_cache_path \
                and os.path.isfile(self.embeddings_cache_path) \
                and self.doc_ids_cache_path \
                and os.path.isfile(self.doc_ids_cache_path):
            logger.info("Đọc dense embeddings từ cache...")
            self._embeddings = np.load(self.embeddings_cache_path)
            with open(self.doc_ids_cache_path, "rb") as f:
                cached_ids = pickle.load(f)
            if len(cached_ids) == len(doc_ids):
                self._doc_ids   = cached_ids
                self._id_to_idx = {did: i for i, did in enumerate(cached_ids)}
                logger.info(
                    f"Loaded embeddings cache: {self._embeddings.shape}"
                )
                return
            else:
                logger.warning(
                    "Cache embeddings kích thước không khớp. Re-encode..."
                )

        self._load_model()
        logger.info(
            f"Encoding {len(raw_texts)} documents "
            f"(batch_size={self.encode_batch_size})..."
        )
        self._embeddings = self._model.encode(
            raw_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize → dot product = cosine
            convert_to_numpy=True,
        ).astype(np.float32)

        logger.info(f"Embeddings shape: {self._embeddings.shape}")

        # Lưu cache
        if self.embeddings_cache_path:
            os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
            np.save(self.embeddings_cache_path, self._embeddings)
            logger.info(f"Saved embeddings cache: {self.embeddings_cache_path}")
        if self.doc_ids_cache_path:
            os.makedirs(os.path.dirname(self.doc_ids_cache_path), exist_ok=True)
            with open(self.doc_ids_cache_path, "wb") as f:
                pickle.dump(self._doc_ids, f)

    # ------------------------------------------------------------------
    # Encode query
    # ------------------------------------------------------------------

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode 1 câu hỏi → vector (D,).
        Dùng văn bản gốc (không tách từ) để phù hợp với tokenizer model.
        """
        self._load_model()
        vec = self._model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def encode_queries_batch(
        self,
        query_texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode nhiều câu hỏi cùng lúc → matrix (Q, D).
        Hiệu quả hơn encode từng câu một khi batch lớn.
        """
        self._load_model()
        logger.info(f"Encoding {len(query_texts)} queries...")
        vecs = self._model.encode(
            query_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # Re-rank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query_vec: np.ndarray,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Tính cosine similarity giữa query_vec và embedding của từng
        candidate_id, sau đó sắp xếp giảm dần.

        Args:
            query_vec     : Vector câu hỏi đã L2-normalize (D,)
            candidate_ids : Danh sách doc_id ứng viên (từ BM25)

        Returns:
            List (doc_id, cosine_score) sắp xếp giảm dần theo score.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Corpus embeddings chưa được tính. Gọi encode_corpus() trước."
            )

        # Lấy indices của candidates
        valid_ids  = [did for did in candidate_ids if did in self._id_to_idx]
        if not valid_ids:
            return []

        indices    = np.array([self._id_to_idx[did] for did in valid_ids])
        cand_embs  = self._embeddings[indices]         # (K, D)

        # Cosine similarity = dot product (embeddings đã L2-normalize)
        scores = cand_embs @ query_vec                 # (K,)

        # Sắp xếp giảm dần
        order  = np.argsort(scores)[::-1]
        return [(valid_ids[i], float(scores[i])) for i in order]

    def batch_rerank(
        self,
        query_vecs: np.ndarray,
        candidates_per_query: List[List[int]],
        show_progress: bool = True,
    ) -> List[List[Tuple[int, float]]]:
        """
        Re-rank nhiều query cùng lúc.

        Args:
            query_vecs           : (Q, D) — kết quả encode_queries_batch
            candidates_per_query : List of candidate_id lists (Q phần tử)

        Returns:
            List (length Q) của List (doc_id, score).
        """
        if len(query_vecs) != len(candidates_per_query):
            raise ValueError("Số query vecs và candidates phải bằng nhau.")

        iterable = tqdm(
            zip(query_vecs, candidates_per_query),
            total=len(query_vecs),
            desc="Dense re-ranking",
            unit="query",
        ) if show_progress else zip(query_vecs, candidates_per_query)

        return [self.rerank(qvec, cands) for qvec, cands in iterable]
