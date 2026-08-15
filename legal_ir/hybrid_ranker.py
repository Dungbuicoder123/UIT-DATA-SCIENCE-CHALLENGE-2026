"""
=============================================================================
LegalIR — Hybrid Ranker & Post-processing
=============================================================================
• Kết hợp BM25 score (sparse) và Dense score để tính Hybrid Score
• Áp dụng threshold và giới hạn số lượng kết quả (≤ 5 IDs/câu hỏi)
• Cung cấp hàm evaluate() mô phỏng Recall@K và Precision@K trên train set
=============================================================================
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def hybrid_score(
    bm25_results: List[Tuple[int, float]],
    dense_results: List[Tuple[int, float]],
    alpha: float = 0.4,
    beta: float = 0.6,
) -> List[Tuple[int, float]]:
    """
    Kết hợp BM25 và Dense score theo công thức linear interpolation:

        hybrid_score = alpha * bm25_norm_score + beta * dense_cosine_score

    Cả hai score đều trong khoảng [0, 1]:
      - bm25 score: đã normalise (min-max) trong BM25Retriever.retrieve()
      - dense score: cosine similarity (đã L2-norm → thuộc [-1, 1],
        thực tế với văn bản tiếng Việt thường ≥ 0)

    Args:
        bm25_results  : [(doc_id, bm25_norm_score), ...]
        dense_results : [(doc_id, cosine_score), ...]
        alpha         : Trọng số BM25
        beta          : Trọng số Dense

    Returns:
        [(doc_id, hybrid_score), ...] sắp xếp giảm dần.
    """
    # Build lookup dict: doc_id → score
    bm25_map  = {did: s for did, s in bm25_results}
    dense_map = {did: s for did, s in dense_results}

    # Tập hợp tất cả doc_id từ cả hai stage
    all_ids = set(bm25_map.keys()) | set(dense_map.keys())

    fused: List[Tuple[int, float]] = []
    for did in all_ids:
        b_score = bm25_map.get(did, 0.0)
        d_score = dense_map.get(did, 0.0)
        h_score = alpha * b_score + beta * d_score
        fused.append((did, h_score))

    # Sắp xếp giảm dần theo hybrid score
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def bm25_only_score(
    bm25_results: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    """Wrapper khi không dùng dense stage."""
    return sorted(bm25_results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Post-processing / selection
# ---------------------------------------------------------------------------

def select_top_docs(
    ranked_results: List[Tuple[int, float]],
    max_docs: int = 5,
    score_threshold: float = 0.0,
) -> List[str]:
    """
    Chọn tối đa `max_docs` văn bản có điểm trên `score_threshold`.

    Ràng buộc quan trọng của cuộc thi: KHÔNG được trả về > 5 IDs.
    Nếu không có kết quả đủ điều kiện, trả về list rỗng.

    Returns:
        Danh sách doc_id (dạng STRING) đã chọn.
    """
    selected = []
    for doc_id, score in ranked_results:
        if len(selected) >= max_docs:
            break
        if score >= score_threshold:
            selected.append(str(doc_id))   # ← xuất dưới dạng string
    return selected


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestrates Stage1 (BM25) + Stage2 (Dense) + post-processing.
    Nhận các retriever đã khởi tạo và chạy full pipeline.
    """

    def __init__(
        self,
        bm25_retriever,         # BM25Retriever instance
        dense_retriever=None,   # DenseRetriever instance (None = BM25-only)
        top_k_stage1: int = 50,
        top_k_stage2: int = 20,
        alpha: float = 0.4,
        beta: float = 0.6,
        max_docs: int = 5,
        score_threshold: float = 0.10,
    ) -> None:
        self.bm25      = bm25_retriever
        self.dense     = dense_retriever
        self.top_k1    = top_k_stage1
        self.top_k2    = top_k_stage2
        self.alpha     = alpha
        self.beta      = beta
        self.max_docs  = max_docs
        self.threshold = score_threshold

    def retrieve_one(
        self,
        processed_query: str,
        raw_query: str,
        query_vec: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], List[Tuple[int, float]]]:
        """
        Chạy pipeline đầy đủ cho 1 câu hỏi.

        Args:
            processed_query : Câu hỏi đã tiền xử lý (cho BM25)
            raw_query       : Câu hỏi gốc (cho Dense encoder)
            query_vec       : Precomputed query vector (tuỳ chọn)

        Returns:
            (selected_doc_ids: List[str], ranked_results: List[(doc_id, score)])
        """
        # Stage 1: BM25
        bm25_results = self.bm25.retrieve(processed_query, top_k=self.top_k1)

        if self.dense is None:
            # BM25-only mode
            ranked = bm25_only_score(bm25_results)
        else:
            # Lấy candidate IDs từ BM25 để re-rank
            candidate_ids = [did for did, _ in bm25_results[:self.top_k2]]

            # Stage 2: Dense re-ranking
            if query_vec is None:
                query_vec = self.dense.encode_query(raw_query)
            dense_results = self.dense.rerank(query_vec, candidate_ids)

            # Kết hợp scores
            ranked = hybrid_score(
                bm25_results[:self.top_k2],
                dense_results,
                alpha=self.alpha,
                beta=self.beta,
            )

        # Post-processing: chọn Top-N với threshold
        selected = select_top_docs(ranked, self.max_docs, self.threshold)
        return selected, ranked

    def batch_retrieve(
        self,
        processed_queries: List[str],
        raw_queries: List[str],
        query_vecs: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> Dict[int, List[str]]:
        """
        Chạy pipeline cho nhiều câu hỏi.

        Args:
            processed_queries : Danh sách câu hỏi đã xử lý (BM25)
            raw_queries       : Danh sách câu hỏi gốc (Dense)
            query_vecs        : Precomputed query vectors (Q, D) — tuỳ chọn
            show_progress     : Hiển thị progress bar

        Returns:
            Dict[query_index → List[str]] (doc IDs dạng string)
        """
        results: Dict[int, List[str]] = {}

        iterable = range(len(processed_queries))
        if show_progress:
            iterable = tqdm(iterable, desc="Hybrid retrieval", unit="query")

        for i in iterable:
            qvec = query_vecs[i] if query_vecs is not None else None
            selected, _ = self.retrieve_one(
                processed_query=processed_queries[i],
                raw_query=raw_queries[i],
                query_vec=qvec,
            )
            results[i] = selected

        return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    predicted: List[str],
    ground_truth: List[str],
    k: int = 5,
) -> float:
    """
    Tính Recall@K cho 1 câu hỏi:
        Recall@K = |predicted[:K] ∩ ground_truth| / |ground_truth|

    Đây là độ đo chính của cuộc thi.
    """
    if not ground_truth:
        return 0.0
    predicted_top_k = set(predicted[:k])
    relevant        = set(ground_truth)
    hits            = len(predicted_top_k & relevant)
    return hits / len(relevant)


def compute_precision_at_k(
    predicted: List[str],
    ground_truth: List[str],
    k: int = 5,
) -> float:
    """
    Tính Precision@K cho 1 câu hỏi:
        Precision@K = |predicted[:K] ∩ ground_truth| / |predicted[:K]|

    Độ đo phụ của cuộc thi.
    """
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    relevant = set(ground_truth)
    hits     = sum(1 for did in top_k if did in relevant)
    return hits / len(top_k)


def compute_f1_at_k(
    predicted: List[str],
    ground_truth: List[str],
    k: int = 5,
) -> float:
    """F1@K = harmonic mean của Recall@K và Precision@K."""
    r = compute_recall_at_k(predicted, ground_truth, k)
    p = compute_precision_at_k(predicted, ground_truth, k)
    if r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)


def evaluate(
    predictions: Dict[str, List[str]],
    ground_truths: Dict[str, List[str]],
    k: int = 5,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Đánh giá toàn bộ tập dự đoán trên train set.

    Args:
        predictions   : {question_id → [predicted_doc_ids]}
        ground_truths : {question_id → [ground_truth_doc_ids]}
        k             : Giới hạn đánh giá (mặc định 5 theo cuộc thi)
        verbose       : In kết quả chi tiết ra console

    Returns:
        Dict chứa mean Recall@K, Precision@K, F1@K,
        và số câu hỏi trả về > k kết quả (vi phạm ràng buộc).
    """
    recall_scores:    List[float] = []
    precision_scores: List[float] = []
    f1_scores:        List[float] = []
    violation_count:  int         = 0   # câu hỏi trả về > 5 IDs

    for qid, gt in ground_truths.items():
        pred = predictions.get(qid, [])

        # Kiểm tra vi phạm ràng buộc
        if len(pred) > k:
            violation_count += 1
            logger.warning(
                f"Query {qid} trả về {len(pred)} IDs > {k} (vi phạm!)"
            )
            pred = []   # Theo quy định cuộc thi: score = 0

        recall_scores.append(compute_recall_at_k(pred, gt, k))
        precision_scores.append(compute_precision_at_k(pred, gt, k))
        f1_scores.append(compute_f1_at_k(pred, gt, k))

    mean_recall    = float(np.mean(recall_scores))    if recall_scores    else 0.0
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mean_f1        = float(np.mean(f1_scores))        if f1_scores        else 0.0

    metrics = {
        f"Recall@{k}":    mean_recall,
        f"Precision@{k}": mean_precision,
        f"F1@{k}":        mean_f1,
        "num_queries":    len(ground_truths),
        "violations":     violation_count,
    }

    if verbose:
        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  Số câu hỏi đánh giá  : {metrics['num_queries']}")
        print(f"  Vi phạm (> {k} IDs)    : {violation_count}")
        print(f"  Recall@{k}             : {mean_recall:.4f}")
        print(f"  Precision@{k}          : {mean_precision:.4f}")
        print(f"  F1@{k}                 : {mean_f1:.4f}")
        print("=" * 55 + "\n")

    return metrics
