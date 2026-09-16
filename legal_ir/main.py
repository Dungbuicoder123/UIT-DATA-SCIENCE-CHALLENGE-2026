"""
LegalIR — Main Pipeline Entry Point
Nâng cấp: Tích hợp Query Expansion + True Hybrid Retrieval + Grid Search Tuning Mode
"""
import argparse
import logging
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_ir.config import (
    CFG,
    BM25_CACHE_PATH,
    CHUNK_MAP_CACHE_PATH,
    CONTEXTS_DIR,
    DOC_IDS_CACHE_PATH,
    EMBEDDINGS_CACHE_PATH,
    OUTPUT_DIR,
    PUBLIC_TEST_FILE,
    TRAIN_FILE,  # ── [ĐÃ THÊM] Import đường dẫn TRAIN_FILE từ config
)
from legal_ir.data_loader import (
    clean_text,
    load_contexts,
    load_queries,
    preprocess,
)
from legal_ir.bm25_retriever import BM25Retriever
from legal_ir.dense_retriever import DenseRetriever
from legal_ir.cross_encoder import CrossEncoderRanker
from legal_ir.hybrid_ranker import (
    HybridRetriever, 
    linear_score_fusion,             # ── [ĐÃ THÊM] Import hàm Linear Fusion
    aggregate_chunks_to_docs,        # ── [ĐÃ THÊM] Dùng cho hàm evaluate tune
    apply_dynamic_threshold          # ── [ĐÃ THÊM] Dùng cho hàm evaluate tune
)
from legal_ir.exporter import export_submission
from legal_ir.query_expander import expand_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_corpus_index(force_rebuild: bool = False):
    """
    Xây dựng toàn bộ index (BM25 + Dense) từ corpus.
    Có cache để tái sử dụng khi chạy lại mà không cần force_rebuild.
    """
    print("═" * 60)
    print("Bước 1: Load Contexts & Chunking (Article-Aware)...")
    print("═" * 60)

    if not force_rebuild and os.path.exists(CHUNK_MAP_CACHE_PATH):
        with open(CHUNK_MAP_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        chunk_ids = cached["cids"]
        raw_texts = cached["raw"]
        bm25_texts = cached["bm25"]
        chunk_map = cached["cmap"]
        print(f"✅ Loaded {len(chunk_ids)} chunks từ cache.")
    else:
        chunk_ids, raw_texts, bm25_texts, chunk_map = load_contexts(
            contexts_dir=CONTEXTS_DIR,
            segmenter=CFG.text.word_segmenter,
            chunk_size=CFG.chunking.chunk_size_words,
            chunk_overlap=CFG.chunking.chunk_overlap_words,
            do_chunking=CFG.chunking.enabled,
            article_aware=CFG.chunking.article_aware,
            inject_header=CFG.chunking.inject_header,
        )
        os.makedirs(os.path.dirname(CHUNK_MAP_CACHE_PATH), exist_ok=True)
        with open(CHUNK_MAP_CACHE_PATH, "wb") as f:
            pickle.dump(
                {"cids": chunk_ids, "raw": raw_texts, "bm25": bm25_texts, "cmap": chunk_map},
                f,
            )
        print(f"✅ Chunked xong: {len(chunk_ids)} chunks từ corpus.")

    print("\nBước 2: Xây dựng BM25 Index...")
    bm25 = BM25Retriever(variant=CFG.bm25.variant, cache_path=BM25_CACHE_PATH)
    bm25.build_index(bm25_texts, chunk_ids, force_rebuild=force_rebuild)
    print(f"✅ BM25 index sẵn sàng (top_k={CFG.bm25.top_k_stage1}).")

    print("\nBước 3: Encode Dense Vectors (Full-Corpus)...")
    dense = None
    if CFG.dense.enabled:
        dense = DenseRetriever(
            model_name=CFG.dense.model_name,
            use_gpu=CFG.dense.use_gpu,
            encode_batch_size=CFG.dense.encode_batch_size,
            max_seq_length=CFG.dense.max_seq_length,
            embeddings_cache_path=EMBEDDINGS_CACHE_PATH,
            doc_ids_cache_path=DOC_IDS_CACHE_PATH,
        )
        dense.encode_corpus(raw_texts, chunk_ids, force_rebuild=force_rebuild)
        print(f"✅ Dense embeddings sẵn sàng (dense_top_k={CFG.dense.dense_top_k}).")
    else:
        print("⚠️  Dense retriever bị tắt (CFG.dense.enabled=False).")

    print("\nBước 4: Khởi tạo Cross-Encoder Reranker...")
    cross = None
    if CFG.cross_encoder.enabled:
        cross = CrossEncoderRanker(
            model_name=CFG.cross_encoder.model_name,
            use_gpu=CFG.cross_encoder.use_gpu,
        )
        print(f"✅ Cross-Encoder sẵn sàng (top_k_stage3={CFG.cross_encoder.top_k_stage3}).")
    else:
        print("⚠️  Cross-Encoder bị tắt (CFG.cross_encoder.enabled=False).")

    raw_texts_dict = {cid: text for cid, text in zip(chunk_ids, raw_texts)}
    return bm25, dense, cross, chunk_map, raw_texts_dict


def run_predict(force_rebuild: bool = False):
    """Pipeline chính: index → query → predict → export."""
    bm25, dense, cross, chunk_map, raw_texts_dict = build_corpus_index(force_rebuild)

    print("\n" + "═" * 60)
    print("Bước 5: Load & Preprocess Queries...")
    print("═" * 60)
    test_queries = load_queries(PUBLIC_TEST_FILE)
    query_ids = [str(q["id"]) for q in test_queries]
    raw_questions = [clean_text(str(q.get("question", ""))) for q in test_queries]

    # Query Expansion: bung từ viết tắt pháp lý trước khi tokenize
    if CFG.text.query_expansion:
        expanded_questions = [expand_query(q) for q in raw_questions]
        print(f"✅ Query Expansion áp dụng cho {len(expanded_questions)} câu hỏi.")
    else:
        expanded_questions = raw_questions

    # Preprocess cho BM25 (sau khi expand)
    processed_queries = [
        preprocess(q, CFG.text.word_segmenter) for q in expanded_questions
    ]

    # Dense encoding dùng raw_questions (model embedding tốt hơn với text tự nhiên)
    # nhưng cũng đã expand để bắt được các từ đầy đủ
    print("\nBước 6: Encode Query Vectors...")
    query_vecs = None
    if dense:
        query_vecs = dense.encode_queries_batch(expanded_questions)
        print(f"✅ Encoded {len(query_vecs)} query vectors.")

    print("\n" + "═" * 60)
    print("Bước 7: 3-Stage Hybrid Retrieval...")
    print("═" * 60)
    retriever = HybridRetriever(bm25, dense, cross, chunk_map, raw_texts_dict, CFG)
    index_to_results = retriever.batch_retrieve(
        processed_queries, expanded_questions, query_vecs
    )

    predictions = {
        query_ids[i]: selected_ids
        for i, selected_ids in index_to_results.items()
    }

    print("\nBước 8: Export Submission...")
    export_submission(query_ids, predictions, OUTPUT_DIR, CFG.hybrid.max_docs_per_query)
    print("\n🎯 Pipeline hoàn tất!")


# ─────────────────────────────────────────────────────────────────────────────
# ── [ĐÃ THÊM] HÀM RUN_TUNE ĐỂ CHẠY GRID SEARCH TỰ ĐỘNG QUÉT ALPHA & BETA ──
# ─────────────────────────────────────────────────────────────────────────────
def run_tune(force_rebuild: bool = False):
    """
    Chạy Grid Search quét trọng số alpha (BM25) và beta (Dense) trên tập train.json
    để tìm tỷ lệ vàng đạt Local Recall@5 cao nhất.
    """
    bm25, dense, cross, chunk_map, raw_texts_dict = build_corpus_index(force_rebuild)

    print("\n" + "═" * 60)
    print("Bước 5 (Tune): Load & Preprocess Validation Queries từ train.json...")
    print("═" * 60)
    
    if not os.path.exists(TRAIN_FILE):
        logger.error(f"Không tìm thấy file train tại: {TRAIN_FILE}")
        return

    queries = load_queries(TRAIN_FILE)
    val_queries = queries[:300]  # Lấy 300 câu đầu làm tập validation cục bộ
    
    query_ids = []
    raw_questions = []
    ground_truths = {}

    for q in val_queries:
        qid = str(q.get("id"))
        q_text = q.get("question") or q.get("query") or ""
        ans = q.get("answer") or q.get("documents") or q.get("doc_ids") or q.get("document_id") or []
        
        if qid and q_text:
            query_ids.append(qid)
            raw_questions.append(clean_text(str(q_text)))
            if isinstance(ans, (list, tuple)):
                ground_truths[qid] = [str(d) for d in ans]
            else:
                ground_truths[qid] = [str(ans)]

    # Query Expansion & Preprocessing
    if CFG.text.query_expansion:
        expanded_questions = [expand_query(q) for q in raw_questions]
    else:
        expanded_questions = raw_questions

    processed_queries = [preprocess(q, CFG.text.word_segmenter) for q in expanded_questions]

    query_vecs = None
    if dense:
        print("Encode query vectors cho validation...")
        query_vecs = dense.encode_queries_batch(expanded_questions)

    print("\nTrích xuất trước kết quả Stage 1 (BM25 & Dense) để tăng tốc quét...")
    bm25_results_all = []
    dense_results_all = []
    for i, p_q in enumerate(tqdm(processed_queries, desc="Stage 1 Pre-retrieval")):
        bm25_res = bm25.retrieve(p_q, top_k=CFG.bm25.top_k_stage1)
        bm25_results_all.append(bm25_res)
        
        dense_res = []
        if dense and query_vecs is not None:
            dense_res = dense.search_full_corpus(query_vecs[i], top_k=CFG.dense.dense_top_k)
        dense_results_all.append(dense_res)

    # Quét Grid Search Alpha từ 0.0 đến 1.0 (step 0.1)
    alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    best_recall = -1.0
    best_alpha = CFG.hybrid.alpha_bm25

    print("\n" + "═" * 60)
    print("Bắt đầu quét Grid Search Alpha (BM25) / Beta (Dense)...")
    print("═" * 60)

    for alpha in alphas:
        beta = round(1.0 - alpha, 1)
        correct_hits = 0
        valid_count = 0

        for i in range(len(query_ids)):
            qid = query_ids[i]
            gt_docs = ground_truths.get(qid, [])
            if not gt_docs:
                continue
            valid_count += 1

            bm25_res = bm25_results_all[i]
            dense_res = dense_results_all[i]

            # Linear Fusion
            if dense_res:
                hybrid_res = linear_score_fusion(bm25_res, dense_res, alpha=alpha, beta=beta)
            else:
                hybrid_res = bm25_res

            # Cross-Encoder Rerank
            if cross and CFG.cross_encoder.enabled:
                top_chunks_for_ce = hybrid_res[: CFG.cross_encoder.top_k_stage3]
                candidates_text = [
                    (cid, raw_texts_dict[cid]) for cid, _ in top_chunks_for_ce if cid in raw_texts_dict
                ]
                reranked = cross.rerank(expanded_questions[i], candidates_text)
            else:
                reranked = hybrid_res

            doc_score_pairs = aggregate_chunks_to_docs(reranked, chunk_map, max_docs=CFG.hybrid.max_docs_per_query)
            
            if CFG.hybrid.dynamic_threshold_enabled and cross and CFG.cross_encoder.enabled:
                selected = apply_dynamic_threshold(
                    doc_score_pairs,
                    min_ce_score=CFG.hybrid.min_ce_score,
                    score_margin=CFG.hybrid.score_margin,
                    max_docs=CFG.hybrid.max_docs_per_query,
                )
            else:
                selected = [str(did) for did, _ in doc_score_pairs]

            # Tính Recall@5
            if any(g in selected for g in gt_docs):
                correct_hits += 1

        recall_score = correct_hits / valid_count if valid_count > 0 else 0.0
        print(f"Alpha (BM25) = {alpha:.1f} | Beta (Dense) = {beta:.1f} ---> Local Recall@5 = {recall_score:.4f}")

        if recall_score > best_recall:
            best_recall = recall_score
            best_alpha = alpha

    print("\n" + "═" * 60)
    print(f"🎯 KẾT QUẢ TỐI ƯU TỪ GRID SEARCH:")
    print(f"alpha_bm25 = {best_alpha:.1f} | beta_dense = {round(1.0 - best_alpha, 1)}")
    print(f"Local Recall@5 cao nhất đạt được: {best_recall:.4f}")
    print("═" * 60)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalIR Prediction Pipeline")
    
    # ── [ĐÃ THÊM] Thêm tham số --mode để chuyển đổi giữa predict và tune ──
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "tune"],
        help="Chế độ chạy: 'predict' để tạo file submission, 'tune' để quét alpha/beta",
    )
    # ─────────────────────────────────────────────────────────────────────

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Xây dựng lại toàn bộ index (bỏ qua cache)",
    )
    args = parser.parse_args()

    # ── [ĐÃ THÊM] Điều hướng gọi hàm tùy thuộc vào mode được truyền vào ──
    if args.mode == "tune":
        run_tune(force_rebuild=args.force_rebuild)
    else:
        run_predict(force_rebuild=args.force_rebuild)
    # ─────────────────────────────────────────────────────────────────────