"""
LegalIR — Hybrid Ranker 
Nâng cấp: Hỗ trợ Linear Score Fusion (Min-Max normalization + Weighted Sum) thay thế/song song RRF
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RRF Fusion (Giữ lại để tương thích ngược nếu cần)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[str, float]], k: int = 60
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _) in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# ★ MỚI: Linear Score Fusion (Min-Max Normalization + Weighted Sum)
# ---------------------------------------------------------------------------

def min_max_normalize(ranked_list: List[Tuple[str, float]]) -> Dict[str, float]:
    """Chuẩn hóa score của một list về khoảng [0, 1] bằng Min-Max Scaling."""
    if not ranked_list:
        return {}
    scores = [score for _, score in ranked_list]
    min_s, max_s = min(scores), max(scores)
    
    # Trường hợp tất cả score bằng nhau
    if max_s == min_s:
        return {cid: 1.0 for cid, _ in ranked_list}
        
    normalized = {}
    for cid, score in ranked_list:
        normalized[cid] = (score - min_s) / (max_s - min_s)
    return normalized


def linear_score_fusion(
    bm25_ranked: List[Tuple[str, float]],
    dense_ranked: List[Tuple[str, float]],
    alpha: float = 0.4,
    beta: float = 0.6,
) -> List[Tuple[str, float]]:
    """
    Kết hợp tuyến tính điểm số giữa BM25 và Dense sau khi đã Min-Max Normalize.
    Final_Score(d) = alpha * Norm_BM25(d) + beta * Norm_Dense(d)
    """
    bm25_norm = min_max_normalize(bm25_ranked)
    dense_norm = min_max_normalize(dense_ranked)
    
    all_cids = set(bm25_norm.keys()).union(set(dense_norm.keys()))
    fused_scores: Dict[str, float] = {}
    
    for cid in all_cids:
        s_bm25 = bm25_norm.get(cid, 0.0)  # Nếu không có trong list, gán 0
        s_dense = dense_norm.get(cid, 0.0)
        
        fused_scores[cid] = alpha * s_bm25 + beta * s_dense
        
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Chunk → Document Aggregation (Max-pooling)
# ---------------------------------------------------------------------------

def aggregate_chunks_to_docs(
    ranked_chunks: List[Tuple[str, float]],
    chunk_to_doc_map: Dict[str, int],
    max_docs: int = 5,
) -> List[Tuple[int, float]]:
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
    if not doc_score_pairs:
        return []

    top_score = doc_score_pairs[0][1]
    selected: List[str] = []

    for doc_id, score in doc_score_pairs[:max_docs]:
        if not selected:
            selected.append(str(doc_id))
            continue
        if score < min_ce_score:
            break
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

        # ── Stage 2: Fusion (RRF hoặc Linear Score Fusion) ─────────────
        if dense_res:
            if cfg_hybrid.fusion_method == "linear":
                hybrid_res = linear_score_fusion(
                    bm25_res, dense_res, 
                    alpha=cfg_hybrid.alpha_bm25, 
                    beta=cfg_hybrid.beta_dense
                )
            else:
                hybrid_res = reciprocal_rank_fusion(
                    bm25_res, dense_res, k=cfg_hybrid.rrf_k
                )
        else:
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
            reranked = hybrid_res

        # ── Aggregate Chunks → Documents ────────────────────────────────
        doc_score_pairs = aggregate_chunks_to_docs(
            reranked, self.chunk_map, max_docs=cfg_hybrid.max_docs_per_query
        )

        # ── Dynamic Threshold / Static Top-K ────────────────────────────
        if cfg_hybrid.dynamic_threshold_enabled and self.cross and cfg_ce.enabled:
            selected = apply_dynamic_threshold(
                doc_score_pairs,
                min_ce_score=cfg_hybrid.min_ce_score,
                score_margin=cfg_hybrid.score_margin,
                max_docs=cfg_hybrid.max_docs_per_query,
            )
        else:
            selected = [str(did) for did, _ in doc_score_pairs]

        return selected

    def batch_retrieve(
        self,
        processed_queries: List[str],
        raw_queries: List[str],
        query_vecs: Optional[np.ndarray],
    ) -> Dict[int, List[str]]:
        from tqdm import tqdm

        results: Dict[int, List[str]] = {}
        for i in tqdm(range(len(processed_queries)), desc="3-Stage Hybrid Retrieval"):
            qvec = query_vecs[i] if query_vecs is not None else None
            results[i] = self.retrieve_one(
                processed_queries[i], raw_queries[i], qvec
            )
        return results