"""
LegalIR — Main Pipeline Entry Point
Nâng cấp: Tích hợp Query Expansion + True Hybrid Retrieval
"""
import argparse
import logging
import os
import pickle
import sys

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
from legal_ir.hybrid_ranker import HybridRetriever
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalIR Prediction Pipeline")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Xây dựng lại toàn bộ index (bỏ qua cache)",
    )
    args = parser.parse_args()
    run_predict(force_rebuild=args.force_rebuild)
