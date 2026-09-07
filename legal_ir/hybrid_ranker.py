"""
LegalIR — Hybrid Ranker (RRF & Max-pooling)
"""
import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
    """Dung hòa xếp hạng bằng công thức RRF: score = 1 / (k + rank)"""
    scores = {}
    for rank, (cid, _) in enumerate(bm25_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(dense_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def aggregate_chunks_to_docs(ranked_chunks: List[Tuple[str, float]], chunk_to_doc_map: Dict[str, int], max_docs: int = 5) -> List[str]:
    """Max-pooling: Điểm của Document = Điểm lớn nhất của các Chunk thuộc Document đó"""
    doc_scores = {}
    for cid, score in ranked_chunks:
        did = chunk_to_doc_map[cid]
        if did not in doc_scores:
            doc_scores[did] = score
        else:
            doc_scores[did] = max(doc_scores[did], score)
            
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [str(did) for did, score in sorted_docs[:max_docs]]

class HybridRetriever:
    def __init__(self, bm25, dense, cross_encoder, chunk_map: Dict[str, int], raw_texts: Dict[str, str], cfg):
        self.bm25, self.dense, self.cross = bm25, dense, cross_encoder
        self.chunk_map, self.raw_texts, self.cfg = chunk_map, raw_texts, cfg

    def retrieve_one(self, processed_query: str, raw_query: str, query_vec: np.ndarray) -> List[str]:
        # Stage 1: BM25 (trên Chunks)
        bm25_res = self.bm25.retrieve(processed_query, top_k=self.cfg.bm25.top_k_stage1)
        
        # Stage 2: Dense (Re-rank BM25 candidates)
        candidate_ids = [cid for cid, _ in bm25_res]
        dense_res = self.dense.rerank(query_vec, candidate_ids)
        
        # Stage 1.5: Fusion bằng RRF
        hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res, k=self.cfg.hybrid.rrf_k)
        
        # Stage 3: Cross-Encoder
        if self.cross and self.cfg.cross_encoder.enabled:
            # Chỉ lấy top K của vòng trước đưa cho Cross-encoder chạy cho lẹ
            top_cids_for_stage3 = hybrid_res[:self.cfg.cross_encoder.top_k_stage3]
            candidates_text = [(cid, self.raw_texts[cid]) for cid, _ in top_cids_for_stage3]
            final_res = self.cross.rerank(raw_query, candidates_text)
        else:
            final_res = hybrid_res

        # Ghép Chunk Score lại thành Document ID
        return aggregate_chunks_to_docs(final_res, self.chunk_map, self.cfg.hybrid.max_docs_per_query)

    def batch_retrieve(self, processed_queries, raw_queries, query_vecs):
        from tqdm import tqdm
        return {i: self.retrieve_one(processed_queries[i], raw_queries[i], query_vecs[i]) 
                for i in tqdm(range(len(processed_queries)), desc="3-Stage Retrieval")}
