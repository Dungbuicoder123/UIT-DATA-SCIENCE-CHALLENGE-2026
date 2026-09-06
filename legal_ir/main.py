"""
=============================================================================
LegalIR — Main Pipeline Entry Point
=============================================================================
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_ir.config import (
    CACHE_DIR, CFG, CONTEXTS_DIR, OUTPUT_DIR,
    PUBLIC_TEST_FILE, SUBMISSION_JSON_PATH, SUBMISSION_ZIP_PATH, TRAIN_FILE,
    BM25_CACHE_PATH, EMBEDDINGS_CACHE_PATH, DOC_IDS_CACHE_PATH,
)
from legal_ir.data_loader import (
    clean_text, load_contexts, load_queries, preprocess,
)
from legal_ir.bm25_retriever import BM25Retriever
from legal_ir.dense_retriever import DenseRetriever
from legal_ir.hybrid_ranker import (
    HybridRetriever, evaluate,
    compute_recall_at_k, compute_precision_at_k,
)
from legal_ir.exporter import export_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build corpus index
# ---------------------------------------------------------------------------
def build_corpus_index(force_rebuild: bool = False) -> tuple:
    # ── 1. Load corpus ───────────────────────────────────────────────────
    # Da bo han duong doc JSONL (load_contexts_fast_jsonl) - ham do khong co
    # chay song song lan cache toan phan, nen khong nhanh hon load_contexts()
    # da toi uu ma con thieu 2 loi ich chinh: tokenize song song (nhanh hon
    # ~so lan bang so core CPU) va cache ket qua cuoi (lan chay sau gan nhu
    # tuc thi neu du lieu khong doi). Dung thang load_contexts() vao thu muc
    # CONTEXTS_DIR (8532 file context_*.json), khong can gop JSONL nua.
    logger.info("Bước 1/3: Đọc corpus văn bản pháp lý...")
    docs, raw_texts, bm25_texts = load_contexts(
        contexts_dir      = CONTEXTS_DIR,
        segmenter         = CFG.text.word_segmenter,
        lowercase         = CFG.text.lowercase,
        name_boost        = CFG.bm25.name_boost,
        checkpoint_path    = os.path.join(CACHE_DIR, "loader_checkpoint.pkl"),
        result_cache_path = os.path.join(CACHE_DIR, "loader_result_cache.pkl"),
    )
    doc_ids = [doc.doc_id for doc in docs]

    # ── 2. BM25 index ────────────────────────────────────────────────────
    logger.info("Bước 2/3: Xây dựng / Load BM25 index từ Drive...")
    bm25 = BM25Retriever(
        variant    = CFG.bm25.variant,
        cache_path = BM25_CACHE_PATH,
    )
    bm25.build_index(bm25_texts, doc_ids, force_rebuild=force_rebuild)

    # ── 3. Dense encoder ─────────────────────────────────────────────────
    dense = None
    if CFG.dense.enabled:
        logger.info("Bước 3/3: Load / Encode Dense embeddings từ Drive...")
        dense = DenseRetriever(
            model_name            = CFG.dense.model_name,
            use_gpu               = CFG.dense.use_gpu,
            encode_batch_size     = CFG.dense.encode_batch_size,
            embeddings_cache_path = EMBEDDINGS_CACHE_PATH,
            doc_ids_cache_path    = DOC_IDS_CACHE_PATH,
        )
        dense.encode_corpus(raw_texts, doc_ids, force_rebuild=force_rebuild)
    else:
        logger.info("Bước 3/3: Bỏ qua Dense encoder (BM25-only mode).")

    return docs, bm25, dense, doc_ids, raw_texts


# ---------------------------------------------------------------------------
# Preprocess queries helper
# ---------------------------------------------------------------------------
def preprocess_queries(queries: List[Dict]) -> tuple:
    query_ids         = []
    processed_queries = []
    raw_queries       = []

    for q in queries:
        qid      = str(q["id"])
        question = str(q.get("question", ""))
        query_ids.append(qid)
        raw_queries.append(clean_text(question))
        processed_queries.append(
            preprocess(question, CFG.text.word_segmenter, CFG.text.lowercase)
        )

    return query_ids, processed_queries, raw_queries


# ---------------------------------------------------------------------------
# Mode: PREDICT
# ---------------------------------------------------------------------------
def run_predict(force_rebuild: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  LegalIR — PREDICT MODE")
    print("=" * 60)

    docs, bm25, dense, doc_ids, raw_texts = build_corpus_index(force_rebuild)

    logger.info(f"Load test queries từ: {PUBLIC_TEST_FILE}")
    test_queries = load_queries(PUBLIC_TEST_FILE)
    query_ids, processed_queries, raw_queries = preprocess_queries(test_queries)

    retriever = HybridRetriever(
        bm25_retriever  = bm25,
        dense_retriever = dense,
        top_k_stage1    = CFG.bm25.top_k_stage1,
        top_k_stage2    = CFG.hybrid.top_k_stage2,
        alpha           = CFG.hybrid.alpha,
        beta            = CFG.hybrid.beta,
        max_docs        = CFG.hybrid.max_docs_per_query,
        score_threshold = CFG.hybrid.score_threshold,
    )

    query_vecs = None
    if dense is not None:
        logger.info("Encode query vectors...")
        query_vecs = dense.encode_queries_batch(raw_queries, show_progress=True)

    logger.info("Chạy Hybrid Retrieval trên tập test...")
    index_to_results = retriever.batch_retrieve(
        processed_queries,
        raw_queries,
        query_vecs=query_vecs,
    )

    predictions: Dict[str, List[str]] = {}
    for idx, selected_ids in index_to_results.items():
        predictions[query_ids[idx]] = selected_ids

    export_submission(
        query_ids   = query_ids,
        predictions = predictions,
        output_dir  = OUTPUT_DIR,
        max_docs    = CFG.hybrid.max_docs_per_query,
    )


# ---------------------------------------------------------------------------
# Mode: EVAL
# ---------------------------------------------------------------------------
def run_eval(force_rebuild: bool = False) -> Dict:
    print("\n" + "=" * 60)
    print("  LegalIR — EVALUATION MODE (train set)")
    print("=" * 60)

    docs, bm25, dense, doc_ids, raw_texts = build_corpus_index(force_rebuild)

    logger.info(f"Load train queries từ: {TRAIN_FILE}")
    train_queries = load_queries(TRAIN_FILE)

    ground_truths: Dict[str, List[str]] = {}
    for q in train_queries:
        qid = str(q["id"])
        answers = q.get("answer", [])
        ground_truths[qid] = [str(a) for a in answers]

    query_ids, processed_queries, raw_queries = preprocess_queries(train_queries)

    retriever = HybridRetriever(
        bm25_retriever  = bm25,
        dense_retriever = dense,
        top_k_stage1    = CFG.bm25.top_k_stage1,
        top_k_stage2    = CFG.hybrid.top_k_stage2,
        alpha           = CFG.hybrid.alpha,
        beta            = CFG.hybrid.beta,
        max_docs        = CFG.hybrid.max_docs_per_query,
        score_threshold = CFG.hybrid.score_threshold,
    )

    query_vecs = None
    if dense is not None:
        query_vecs = dense.encode_queries_batch(raw_queries, show_progress=True)

    index_to_results = retriever.batch_retrieve(
        processed_queries,
        raw_queries,
        query_vecs=query_vecs,
    )

    predictions: Dict[str, List[str]] = {}
    for idx, selected_ids in index_to_results.items():
        predictions[query_ids[idx]] = selected_ids

    metrics = evaluate(
        predictions   = predictions,
        ground_truths = ground_truths,
        k             = CFG.hybrid.max_docs_per_query,
        verbose       = True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": {
                    "bm25_variant":    CFG.bm25.variant,
                    "top_k_stage1":    CFG.bm25.top_k_stage1,
                    "dense_model":     CFG.dense.model_name,
                    "dense_enabled":   CFG.dense.enabled,
                    "alpha":           CFG.hybrid.alpha,
                    "beta":            CFG.hybrid.beta,
                    "score_threshold": CFG.hybrid.score_threshold,
                    "top_k_stage2":    CFG.hybrid.top_k_stage2,
                    "max_docs":        CFG.hybrid.max_docs_per_query,
                    "word_segmenter":  CFG.text.word_segmenter,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Kết quả đánh giá đã lưu tại: {eval_path}")
    return metrics


# ---------------------------------------------------------------------------
# Mode: TUNE
# ---------------------------------------------------------------------------
def run_tune(force_rebuild: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  LegalIR — HYPERPARAMETER TUNING")
    print("=" * 60)

    import itertools

    docs, bm25, dense, doc_ids, raw_texts = build_corpus_index(force_rebuild)

    train_queries = load_queries(TRAIN_FILE)
    ground_truths = {
        str(q["id"]): [str(a) for a in q.get("answer", [])]
        for q in train_queries
    }
    query_ids, processed_queries, raw_queries = preprocess_queries(train_queries)

    query_vecs = None
    if dense is not None:
        query_vecs = dense.encode_queries_batch(raw_queries, show_progress=False)

    alpha_values        = [0.2, 0.4, 0.6]
    threshold_values    = [0.05, 0.10, 0.15, 0.20]
    top_k_stage1_values = [30, 50, 100]

    best_recall = -1.0
    best_config = {}
    results_table = []

    total = len(alpha_values) * len(threshold_values) * len(top_k_stage1_values)
    logger.info(f"Grid search: {total} tổ hợp hyperparameters...")

    for alpha, threshold, top_k1 in tqdm(
        itertools.product(alpha_values, threshold_values, top_k_stage1_values),
        total=total,
        desc="Tuning",
    ):
        beta = 1.0 - alpha
        retriever = HybridRetriever(
            bm25_retriever  = bm25,
            dense_retriever = dense,
            top_k_stage1    = top_k1,
            top_k_stage2    = min(20, top_k1),
            alpha           = alpha,
            beta            = beta,
            max_docs        = 5,
            score_threshold = threshold,
        )

        idx_results = retriever.batch_retrieve(
            processed_queries,
            raw_queries,
            query_vecs=query_vecs,
            show_progress=False,
        )
        preds = {query_ids[i]: v for i, v in idx_results.items()}
        metrics = evaluate(preds, ground_truths, k=5, verbose=False)

        recall = metrics["Recall@5"]
        row = {
            "alpha": alpha, "beta": beta,
            "threshold": threshold, "top_k1": top_k1,
            "Recall@5": recall,
            "Precision@5": metrics["Precision@5"],
            "F1@5": metrics["F1@5"],
        }
        results_table.append(row)

        if recall > best_recall:
            best_recall = recall
            best_config = row

    tune_path = os.path.join(OUTPUT_DIR, "tune_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best": best_config, "all": results_table},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n🏆 Best config (Recall@5 = {best_recall:.4f}):")
    for k, v in best_config.items():
        print(f"   {k}: {v}")
    print(f"\n📁 Kết quả đầy đủ lưu tại: {tune_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LegalIR — Hệ thống truy xuất văn bản pháp lý tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "eval", "tune"],
        default="predict",
    )
    parser.add_argument(
        "--no-dense",
        action="store_true",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_dense:
        CFG.dense.enabled = False
        logger.info("Dense retriever ĐÃ BỊ TẮT (BM25-only mode).")

    if args.alpha is not None:
        CFG.hybrid.alpha = args.alpha
        CFG.hybrid.beta  = 1.0 - args.alpha

    if args.threshold is not None:
        CFG.hybrid.score_threshold = args.threshold

    if args.mode == "predict":
        run_predict(force_rebuild=args.force_rebuild)
    elif args.mode == "eval":
        run_eval(force_rebuild=args.force_rebuild)
    elif args.mode == "tune":
        run_tune(force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()