"""
LegalIR Configuration
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

@dataclass
class ChunkingConfig:
    enabled: bool = True
    chunk_size_words: int = 256
    chunk_overlap_words: int = 64

@dataclass
class BM25Config:
    top_k_stage1: int = 100  # Tăng lên 100 để lấy nhiều ứng viên (chunk) hơn cho RRF
    variant: str = "BM25Okapi"
    name_boost: float = 2.0

@dataclass
class DenseConfig:
    model_name: str = "AITeamVN/Vietnamese_Embedding"
    encode_batch_size: int = 512
    use_gpu: bool = True
    enabled: bool = True
    max_seq_length: int = 2048

@dataclass
class CrossEncoderConfig:
    enabled: bool = True
    model_name: str = "AITeamVN/Vietnamese_Reranker"
    use_gpu: bool = True
    top_k_stage3: int = 20  # Lấy top 20 chunk cao điểm nhất từ RRF để rerank

@dataclass
class HybridConfig:
    fusion_method: str = "rrf"  # Dùng RRF
    rrf_k: int = 60
    max_docs_per_query: int = 5

@dataclass
class Config:
    text: TextConfig = field(default_factory=TextConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

CFG = Config()
