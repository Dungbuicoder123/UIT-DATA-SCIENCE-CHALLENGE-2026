"""
LegalIR — Main Pipeline Entry Point (Đã Nâng Cấp 3 Stages)
"""
import argparse, os, pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_ir.config import CFG, CONTEXTS_DIR, BM25_CACHE_PATH, EMBEDDINGS_CACHE_PATH, DOC_IDS_CACHE_PATH, CHUNK_MAP_CACHE_PATH, PUBLIC_TEST_FILE, OUTPUT_DIR
from legal_ir.data_loader import load_contexts, load_queries, preprocess, clean_text
from legal_ir.bm25_retriever import BM25Retriever
from legal_ir.dense_retriever import DenseRetriever
from legal_ir.cross_encoder import CrossEncoderRanker
from legal_ir.hybrid_ranker import HybridRetriever
from legal_ir.exporter import export_submission

def build_corpus_index(force_rebuild=False):
    print("Bước 1: Load Contexts & Semantic Chunking...")
    if not force_rebuild and os.path.exists(CHUNK_MAP_CACHE_PATH):
        with open(CHUNK_MAP_CACHE_PATH, "rb") as f: cached = pickle.load(f)
        chunk_ids, raw_texts, bm25_texts, chunk_map = cached["cids"], cached["raw"], cached["bm25"], cached["cmap"]
    else:
        chunk_ids, raw_texts, bm25_texts, chunk_map = load_contexts(
            contexts_dir=CONTEXTS_DIR, segmenter=CFG.text.word_segmenter, 
            chunk_size=CFG.chunking.chunk_size_words, chunk_overlap=CFG.chunking.chunk_overlap_words, do_chunking=CFG.chunking.enabled
        )
        os.makedirs(os.path.dirname(CHUNK_MAP_CACHE_PATH), exist_ok=True)
        with open(CHUNK_MAP_CACHE_PATH, "wb") as f: pickle.dump({"cids": chunk_ids, "raw": raw_texts, "bm25": bm25_texts, "cmap": chunk_map}, f)

    print("Bước 2: Xây dựng BM25 Index (trên Chunks)...")
    bm25 = BM25Retriever(variant=CFG.bm25.variant, cache_path=BM25_CACHE_PATH)
    bm25.build_index(bm25_texts, chunk_ids, force_rebuild=force_rebuild)

    print("Bước 3: Encode Dense Vector (trên Chunks)...")
    dense = None
    if CFG.dense.enabled:
        dense = DenseRetriever(CFG.dense.model_name, CFG.dense.use_gpu, CFG.dense.encode_batch_size, CFG.dense.max_seq_length, EMBEDDINGS_CACHE_PATH, DOC_IDS_CACHE_PATH)
        dense.encode_corpus(raw_texts, chunk_ids, force_rebuild=force_rebuild)

    print("Bước 4: Chuẩn bị Cross-Encoder...")
    cross = CrossEncoderRanker(CFG.cross_encoder.model_name, CFG.cross_encoder.use_gpu) if CFG.cross_encoder.enabled else None

    raw_texts_dict = {cid: text for cid, text in zip(chunk_ids, raw_texts)}
    return bm25, dense, cross, chunk_map, raw_texts_dict

def run_predict(force_rebuild=False):
    bm25, dense, cross, chunk_map, raw_texts_dict = build_corpus_index(force_rebuild)
    
    test_queries = load_queries(PUBLIC_TEST_FILE)
    query_ids = [str(q["id"]) for q in test_queries]
    processed_queries = [preprocess(str(q.get("question", "")), CFG.text.word_segmenter) for q in test_queries]
    raw_queries = [clean_text(str(q.get("question", ""))) for q in test_queries]
    
    query_vecs = dense.encode_queries_batch(raw_queries) if dense else None
    
    retriever = HybridRetriever(bm25, dense, cross, chunk_map, raw_texts_dict, CFG)
    index_to_results = retriever.batch_retrieve(processed_queries, raw_queries, query_vecs)
    
    predictions = {query_ids[i]: selected_ids for i, selected_ids in index_to_results.items()}
    export_submission(query_ids, predictions, OUTPUT_DIR, CFG.hybrid.max_docs_per_query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    run_predict(force_rebuild=args.force_rebuild)
