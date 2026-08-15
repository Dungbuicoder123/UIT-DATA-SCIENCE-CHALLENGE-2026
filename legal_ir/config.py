"""
=============================================================================
LegalIR Configuration
=============================================================================
Centralised configuration for the entire LegalIR pipeline.
Adjust paths, hyperparameters and model choices here before running.
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Directory / file paths  (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR              = os.path.join(BASE_DIR, "data")
CONTEXTS_DIR          = os.path.join(DATA_DIR, "selected-contexts")   # unzipped folder
TRAIN_FILE            = os.path.join(DATA_DIR, "train.json")
PUBLIC_TEST_FILE      = os.path.join(DATA_DIR, "public-official.json")
OUTPUT_DIR            = os.path.join(BASE_DIR, "output")
SUBMISSION_JSON_PATH  = os.path.join(OUTPUT_DIR, "submission.json")
SUBMISSION_ZIP_PATH   = os.path.join(OUTPUT_DIR, "submission.zip")

# Cache paths (avoid recomputing heavy embeddings every run)
CACHE_DIR             = os.path.join(BASE_DIR, ".cache")
CORPUS_CACHE_PATH     = os.path.join(CACHE_DIR, "preprocessed_corpus.pkl")
BM25_CACHE_PATH       = os.path.join(CACHE_DIR, "bm25_index.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "corpus_embeddings.npy")
DOC_IDS_CACHE_PATH    = os.path.join(CACHE_DIR, "doc_ids.pkl")


# ---------------------------------------------------------------------------
# BM25 hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class BM25Config:
    # Number of candidate documents retrieved in Stage-1
    top_k_stage1: int = 50

    # BM25 variant: "BM25Okapi" | "BM25L" | "BM25Plus"
    variant: str = "BM25Okapi"

    # Combine passage + name in BM25 index (gives extra weight to doc title)
    use_name_in_index: bool = True

    # Weight multiplier for tokens coming from the document *name* field
    name_boost: float = 2.0


# ---------------------------------------------------------------------------
# Dense retrieval / re-ranking hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class DenseConfig:
    # HuggingFace model ID (Vietnamese bi-encoder)
    # Alternatives:
    #   "bkai-foundation-models/vietnamese-bi-encoder"
    #   "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    #   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"

    # Batch size for encoding (lower if GPU/RAM is limited)
    encode_batch_size: int = 64

    # Number of threads for CPU encoding
    num_threads: int = 4

    # Use GPU if available
    use_gpu: bool = True

    # Whether to enable dense stage at all (set False for BM25-only mode)
    enabled: bool = True


# ---------------------------------------------------------------------------
# Hybrid retrieval & final selection hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class HybridConfig:
    # Weight for BM25 normalised score  [0, 1]
    alpha: float = 0.4

    # Weight for dense cosine score   [0, 1]   (alpha + beta should == 1.0)
    beta: float = 0.6

    # Hard upper limit per query (contest constraint: max 5 IDs)
    max_docs_per_query: int = 5

    # Minimum hybrid score threshold to include a doc.
    # Docs below this threshold are excluded even if < max_docs_per_query.
    score_threshold: float = 0.10

    # Number of dense-stage candidates (must be <= top_k_stage1)
    top_k_stage2: int = 20


# ---------------------------------------------------------------------------
# Text processing settings
# ---------------------------------------------------------------------------
@dataclass
class TextConfig:
    # Word segmentation backend: "pyvi" (fastest, ~1s) | "underthesea" | "none"
    word_segmenter: str = "pyvi"

    # Whether to lower-case text
    lowercase: bool = True

    # Whether to strip Vietnamese diacritics during BM25 indexing
    # (can improve recall for queries without diacritics)
    strip_accents_for_bm25: bool = False


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    # Maximum number of returned docs evaluated (contest constraint)
    max_retrieved: int = 5


# ---------------------------------------------------------------------------
# Master config (compose all sub-configs)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    bm25: BM25Config   = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    text: TextConfig   = field(default_factory=TextConfig)
    eval: EvalConfig   = field(default_factory=EvalConfig)


# Singleton instance used throughout the project
CFG = Config()
