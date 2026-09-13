"""
LegalIR — Hybrid Ranker
Nâng cấp:
  1. True Full-Corpus Dense Search (song song với BM25, không phụ thuộc vào nhau)
  2. RRF Fusion trên union của 2 candidate lists
  3. Dynamic Margin Thresholding sau Cross-Encoder (tối ưu Precision)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[str, float]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Dung hòa nhiều ranked list bằng Reciprocal Rank Fusion.

    Score_RRF(d) = Σ_list  1 / (k + rank(d, list))

    Nhận *ranked_lists để hỗ trợ bất kỳ số list nào (BM25, Dense, ...).
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _) in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Chunk → Document Aggregation (Max-pooling)
# ---------------------------------------------------------------------------

def aggregate_chunks_to_docs(
    ranked_chunks: List[Tuple[str, float]],
    chunk_to_doc_map: Dict[str, int],
    max_docs: int = 5,
) -> List[Tuple[int, float]]:
    """
    Max-pooling: điểm của Document = điểm lớn nhất trong các chunk của nó.

    Trả về: List[(doc_id, score)] đã sắp xếp giảm dần, tối đa max_docs.
    """
    doc_scores: Dict[int, float] = {}
    for cid, score in ranked_chunks:
        did = chunk_to_doc_map.get(cid)
        if did is None:
            continue
        if did not in doc_scores or score > doc_scores[did]:
            doc_scores[did] = score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:max_docs]


# ---------------------------------------------------------------------------
# Dynamic Margin Thresholding
# ---------------------------------------------------------------------------

def apply_dynamic_threshold(
    doc_score_pairs: List[Tuple[int, float]],
    min_ce_score: float = -3.0,
    score_margin: float = 4.5,
    max_docs: int = 5,
) -> List[str]:
    """
    Lọc danh sách doc_ids dựa trên điểm Cross-Encoder:

    Giữ doc_i nếu:
      1. score_i >= min_ce_score  (ngưỡng tuyệt đối tối thiểu)
      2. score_top1 - score_i <= score_margin  (không quá xa top-1)

    Luôn giữ ít nhất 1 document (top-1).
    Không bao giờ vượt max_docs (ràng buộc cuộc thi).

    Returns:
        List[str] — danh sách doc_id (string) đã lọc.
    """
    if not doc_score_pairs:
        return []

    top_score = doc_score_pairs[0][1]
    selected: List[str] = []

    for doc_id, score in doc_score_pairs[:max_docs]:
        # Luôn giữ top-1
        if not selected:
            selected.append(str(doc_id))
            continue
        # Kiểm tra ngưỡng tuyệt đối
        if score < min_ce_score:
            break
        # Kiểm tra khoảng cách so với top-1
        if (top_score - score) > score_margin:
            break
        selected.append(str(doc_id))

    return selected


# ---------------------------------------------------------------------------
# HybridRetriever — Pipeline chính
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        bm25,
        dense,
        cross_encoder,
        chunk_map: Dict[str, int],
        raw_texts: Dict[str, str],
        cfg,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.cross = cross_encoder
        self.chunk_map = chunk_map
        self.raw_texts = raw_texts
        self.cfg = cfg

    def retrieve_one(
        self,
        processed_query: str,
        raw_query: str,
        query_vec: Optional[np.ndarray],
    ) -> List[str]:
        """
        3-Stage True Hybrid Retrieval cho 1 câu hỏi:

        Stage 1 — Candidate Generation (song song, độc lập):
          ├─ BM25 Full-Corpus Search → Top bm25_k chunks
          └─ Dense Full-Corpus Search → Top dense_k chunks  ← MỚI

        Stage 2 — RRF Fusion:
          BM25_list ⊕ Dense_list → Top rrf_out chunks

        Stage 3 — Cross-Encoder Rerank:
          Top cross_k chunks → Cross-Encoder score → Sorted list

        Post-processing:
          Max-pool chunks → doc scores → Dynamic Threshold → 1~5 doc_ids
        """
        cfg_bm25 = self.cfg.bm25
        cfg_dense = self.cfg.dense
        cfg_ce = self.cfg.cross_encoder
        cfg_hybrid = self.cfg.hybrid

        # ── Stage 1A: BM25 Full-Corpus Search ──────────────────────────
        bm25_res: List[Tuple[str, float]] = self.bm25.retrieve(
            processed_query, top_k=cfg_bm25.top_k_stage1
        )

        # ── Stage 1B: Dense Full-Corpus Search ─────────────────────────
        dense_res: List[Tuple[str, float]] = []
        if self.dense and cfg_dense.enabled and query_vec is not None:
            dense_res = self.dense.search_full_corpus(
                query_vec, top_k=cfg_dense.dense_top_k
            )

        # ── Stage 2: RRF Fusion ─────────────────────────────────────────
        if dense_res:
            hybrid_res = reciprocal_rank_fusion(
                bm25_res, dense_res, k=cfg_hybrid.rrf_k
            )
        else:
            # Fallback: chỉ dùng BM25 nếu Dense không khả dụng
            hybrid_res = bm25_res

        # ── Stage 3: Cross-Encoder Rerank ──────────────────────────────
        if self.cross and cfg_ce.enabled:
            top_chunks_for_ce = hybrid_res[: cfg_ce.top_k_stage3]
            candidates_text = [
                (cid, self.raw_texts[cid])
                for cid, _ in top_chunks_for_ce
                if cid in self.raw_texts
            ]
            reranked = self.cross.rerank(raw_query, candidates_text)
        else:
            # Không có Cross-Encoder: chuyển hybrid_res sang định dạng tương tự
            reranked = hybrid_res

        # ── Aggregate Chunks → Documents ────────────────────────────────
        doc_score_pairs = aggregate_chunks_to_docs(
            reranked, self.chunk_map, max_docs=cfg_hybrid.max_docs_per_query
        )

        # ── Dynamic Threshold / Static Top-K ────────────────────────────
        if cfg_hybrid.dynamic_threshold_enabled and self.cross and cfg_ce.enabled:
            # Khi có Cross-Encoder: dùng dynamic margin filtering
            selected = apply_dynamic_threshold(
                doc_score_pairs,
                min_ce_score=cfg_hybrid.min_ce_score,
                score_margin=cfg_hybrid.score_margin,
                max_docs=cfg_hybrid.max_docs_per_query,
            )
        else:
            # Không có Cross-Encoder hoặc tắt dynamic: lấy cố định Top-K
            selected = [str(did) for did, _ in doc_score_pairs]

        return selected

    def batch_retrieve(
        self,
        processed_queries: List[str],
        raw_queries: List[str],
        query_vecs: Optional[np.ndarray],
    ) -> Dict[int, List[str]]:
        """Chạy retrieve_one cho toàn bộ batch queries."""
        from tqdm import tqdm

        results: Dict[int, List[str]] = {}
        for i in tqdm(range(len(processed_queries)), desc="3-Stage Hybrid Retrieval"):
            qvec = query_vecs[i] if query_vecs is not None else None
            results[i] = self.retrieve_one(
                processed_queries[i], raw_queries[i], qvec
            )
        return results
