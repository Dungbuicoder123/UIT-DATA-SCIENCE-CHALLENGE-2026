"""
=============================================================================
LegalIR — Stage 1: BM25 Sparse Retriever
=============================================================================
• Xây dựng BM25 index từ corpus passages
• Hỗ trợ cache index để không phải build lại mỗi lần chạy
• Cung cấp hàm `retrieve` trả về Top-K kết quả với normalised score
=============================================================================
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Map tên variant → class BM25
_BM25_VARIANTS = {
    "BM25Okapi": BM25Okapi,
    "BM25L":     BM25L,
    "BM25Plus":  BM25Plus,
}


class BM25Retriever:
    """
    Sparse retriever dựa trên BM25, hỗ trợ:
      - Ba biến thể: BM25Okapi, BM25L, BM25Plus
      - Cache index ra đĩa (pickle)
      - Normalised score (min-max → [0, 1]) để dễ kết hợp với Dense score
    """

    def __init__(
        self,
        variant: str = "BM25Okapi",
        cache_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            variant    : Tên biến thể BM25 ("BM25Okapi" | "BM25L" | "BM25Plus")
            cache_path : Đường dẫn lưu/đọc cache pickle. None = không cache.
        """
        if variant not in _BM25_VARIANTS:
            raise ValueError(
                f"variant phải là một trong {list(_BM25_VARIANTS.keys())}"
            )
        self.variant    = variant
        self.cache_path = cache_path
        self._bm25      = None          # BM25 index object
        self._doc_ids: List[int] = []   # mapping index → doc_id

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build_index(
        self,
        bm25_texts: List[str],
        doc_ids: List[int],
        force_rebuild: bool = False,
    ) -> None:
        """
        Xây dựng BM25 index từ danh sách văn bản đã tiền xử lý.

        Args:
            bm25_texts   : Danh sách chuỗi văn bản đã tách từ (parallel với doc_ids)
            doc_ids      : Danh sách doc_id tương ứng
            force_rebuild: Bỏ qua cache, build lại từ đầu
        """
        if len(bm25_texts) != len(doc_ids):
            raise ValueError("bm25_texts và doc_ids phải có cùng độ dài.")

        self._doc_ids = doc_ids

        # Thử load từ cache
        if not force_rebuild and self.cache_path and \
                os.path.isfile(self.cache_path):
            logger.info(f"Đọc BM25 index từ cache: {self.cache_path}")
            self._load_cache()
            # Kiểm tra kích thước khớp không
            if len(self._doc_ids) == len(doc_ids):
                return
            else:
                logger.warning(
                    "Cache có kích thước khác corpus. Rebuild index..."
                )

        # Tokenize (tách token theo khoảng trắng —
        # underthesea/pyvi đã tách từ bằng "_", nên split(" ") là đủ)
        logger.info(f"Tokenizing {len(bm25_texts)} documents cho BM25...")
        tokenized = [text.split() for text in tqdm(bm25_texts, desc="Tokenize")]

        logger.info(f"Xây dựng BM25 ({self.variant}) index...")
        bm25_cls = _BM25_VARIANTS[self.variant]
        self._bm25 = bm25_cls(tokenized)
        logger.info("Build BM25 index hoàn thành.")

        # Lưu cache
        if self.cache_path:
            self._save_cache()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Tìm kiếm Top-K văn bản cho một câu hỏi.

        Args:
            query: Câu hỏi đã tiền xử lý (đã tách từ)
            top_k: Số lượng kết quả trả về

        Returns:
            Danh sách (doc_id, normalised_score) sắp xếp giảm dần theo score.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "Index chưa được build. Gọi build_index() trước."
            )

        query_tokens = query.split()
        scores = self._bm25.get_scores(query_tokens)  # ndarray length = corpus

        # Lấy Top-K index
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Normalise scores sang [0, 1] (min-max trên top_k kết quả)
        top_scores = scores[top_indices]
        s_min, s_max = top_scores.min(), top_scores.max()
        if s_max > s_min:
            norm_scores = (top_scores - s_min) / (s_max - s_min)
        else:
            # Tất cả điểm bằng nhau (thường khi query không khớp gì)
            norm_scores = np.zeros_like(top_scores)

        results: List[Tuple[int, float]] = []
        for idx, norm_score in zip(top_indices, norm_scores):
            doc_id = self._doc_ids[idx]
            results.append((doc_id, float(norm_score)))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 50,
        show_progress: bool = True,
    ) -> List[List[Tuple[int, float]]]:
        """
        Retrieve cho nhiều câu hỏi cùng lúc.

        Returns:
            Danh sách kết quả, mỗi phần tử tương ứng với 1 query.
        """
        iterable = tqdm(queries, desc="BM25 retrieval", unit="query") \
                   if show_progress else queries
        return [self.retrieve(q, top_k) for q in iterable]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "doc_ids": self._doc_ids},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"Đã lưu BM25 index cache: {self.cache_path}")

    def _load_cache(self) -> None:
        with open(self.cache_path, "rb") as f:
            cached = pickle.load(f)
        self._bm25    = cached["bm25"]
        self._doc_ids = cached["doc_ids"]
        logger.info(
            f"Đã load BM25 index cache ({len(self._doc_ids)} docs)."
        )
