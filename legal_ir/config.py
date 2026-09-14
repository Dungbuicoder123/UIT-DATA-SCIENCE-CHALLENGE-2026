"""
LegalIR Configuration — Bản chuẩn tích hợp Linear Fusion & Đầy đủ thuộc tính
"""
import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONTEXTS_DIR = os.path.join(DATA_DIR, "selected-contexts")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
PUBLIC_TEST_FILE = os.path.join(DATA_DIR, "public-official.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBMISSION_JSON_PATH = os.path.join(OUTPUT_DIR, "submission.json")
SUBMISSION_ZIP_PATH = os.path.join(OUTPUT_DIR, "submission.zip")
CACHE_DIR = "/content/drive/MyDrive/LegalIR_Project/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache paths
BM25_CACHE_PATH = os.path.join(CACHE_DIR, "bm25_index.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "corpus_embeddings.npy")
DOC_IDS_CACHE_PATH = os.path.join(CACHE_DIR, "doc_ids.pkl")
CHUNK_MAP_CACHE_PATH = os.path.join(CACHE_DIR, "chunk_map.pkl")

@dataclass
class TextConfig:
    word_segmenter: str = "pyvi"
    lowercase: bool = True
    query_expansion: bool = True

@dataclass
class ChunkingConfig:
    enabled: bool = True
    chunk_size_words: int = 256
    chunk_overlap_words: int = 64
    article_aware: bool = True
    inject_header: bool = True

@dataclass
class BM25Config:
    top_k_stage1: int = 150
    variant: str = "BM25Okapi"
    name_boost: float = 2.0

@dataclass
class DenseConfig:
    model_name: str = "AITeamVN/Vietnamese_Embedding"
    encode_batch_size: int = 128
    use_gpu: bool = True
    enabled: bool = True
    max_seq_length: int = 2048
    # Khôi phục đầy đủ biến dense_top_k để tránh lỗi AttributeError
    dense_top_k: int = 150

@dataclass
class CrossEncoderConfig:
    enabled: bool = True
    model_name: str = "AITeamVN/Vietnamese_Reranker"
    use_gpu: bool = True
    top_k_stage3: int = 30

@dataclass
class HybridConfig:
    # Chọn phương thức "linear" để chạy Linear Score Fusion
    fusion_method: str = "linear"  
    rrf_k: int = 60
    
    # Trọng số tuyến tính cho BM25 và Dense
    alpha_bm25: float = 0.4      
    beta_dense: float = 0.6      
    
    max_docs_per_query: int = 5

    dynamic_threshold_enabled: bool = True
    min_ce_score: float = -3.0
    score_margin: float = 4.5

@dataclass
class Config:
    text: TextConfig = field(default_factory=TextConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

CFG = Config()