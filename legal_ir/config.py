"""
=============================================================================
LegalIR Configuration
=============================================================================
Centralised configuration for the entire LegalIR pipeline.
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Directory / file paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR              = os.path.join(BASE_DIR, "data")
CONTEXTS_DIR          = os.path.join(DATA_DIR, "selected-contexts")
TRAIN_FILE            = os.path.join(DATA_DIR, "train.json")
PUBLIC_TEST_FILE      = os.path.join(DATA_DIR, "public-official.json")
OUTPUT_DIR            = os.path.join(BASE_DIR, "output")
SUBMISSION_JSON_PATH  = os.path.join(OUTPUT_DIR, "submission.json")
SUBMISSION_ZIP_PATH   = os.path.join(OUTPUT_DIR, "submission.zip")

# [QUAN TRỌNG] Lưu cache thẳng vào Drive thay vì thư mục tạm .cache của Colab
CACHE_DIR             = "/content/drive/MyDrive/LegalIR_Project/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BM25_CACHE_PATH       = os.path.join(CACHE_DIR, "bm25_index.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "corpus_embeddings.npy")
DOC_IDS_CACHE_PATH    = os.path.join(CACHE_DIR, "doc_ids.pkl")


# ---------------------------------------------------------------------------
# BM25 hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class BM25Config:
    top_k_stage1: int = 50
    variant: str = "BM25Okapi"
    use_name_in_index: bool = True
    name_boost: float = 2.0


# ---------------------------------------------------------------------------
# Dense retrieval / re-ranking hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class DenseConfig:
    model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"
    # Tăng batch size lên 128 giúp tận dụng tốt GPU T4 của Colab
    encode_batch_size: int = 128
    num_threads: int = 4
    use_gpu: bool = True
    enabled: bool = True


# ---------------------------------------------------------------------------
# Hybrid retrieval & final selection hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class HybridConfig:
    alpha: float = 0.4
    beta: float = 0.6
    max_docs_per_query: int = 5
    score_threshold: float = 0.10
    top_k_stage2: int = 20


# ---------------------------------------------------------------------------
# Text processing settings
# ---------------------------------------------------------------------------
@dataclass
class TextConfig:
    word_segmenter: str = "underthesea"
    lowercase: bool = True
    strip_accents_for_bm25: bool = False


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    max_retrieved: int = 5


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    bm25: BM25Config   = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    text: TextConfig   = field(default_factory=TextConfig)
    eval: EvalConfig   = field(default_factory=EvalConfig)


CFG = Config()