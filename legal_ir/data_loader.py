"""
=============================================================================
LegalIR — Data Loader & Text Preprocessor (High-Performance Parallel Edition)
=============================================================================
Handles:
  • Loading and parsing context JSON files from selected-contexts/
  • Multiprocessing parallel processing across all CPU cores
  • Automatic disk caching of preprocessed corpus (.cache/preprocessed_corpus.pkl)
  • Fast Vietnamese text cleaning, Unicode normalisation, word segmentation
=============================================================================
"""

import glob
import json
import logging
import os
import pickle
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vietnamese text utilities
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Chuẩn hoá Unicode về dạng NFC (khắc phục lỗi font tiếng Việt tổ hợp).
    VD: chữ 'à' được tổ hợp từ 'a' + dấu huyền → gộp thành 1 ký tự.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản tiếng Việt:
      1. Normalize Unicode (NFC)
      2. Bỏ ký tự điều khiển, thay whitespace dư thừa bằng dấu cách
      3. Giữ nguyên dấu câu cơ bản để không mất ngữ nghĩa pháp lý
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    # Loại ký tự điều khiển (ASCII 0-8, 11-12, 14-31)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Thay nhiều khoảng trắng / tab / newline bằng 1 dấu cách
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_words(text: str, segmenter: str = "pyvi") -> str:
    """
    Tách từ tiếng Việt.
    - "pyvi"        : dùng thư viện pyvi (siêu nhanh, ~100x hơn underthesea)
    - "underthesea" : dùng thư viện underthesea
    - "none"        : không tách từ

    Trả về chuỗi đã tách từ (từ ghép được nối bằng dấu gạch dưới: _).
    """
    if not text:
        return ""

    if segmenter == "pyvi":
        try:
            from pyvi import ViTokenizer  # type: ignore
            return ViTokenizer.tokenize(text)
        except ImportError:
            segmenter = "underthesea"

    if segmenter == "underthesea":
        try:
            from underthesea import word_tokenize  # type: ignore
            return word_tokenize(text, format="text")
        except ImportError:
            return text

    # "none" — trả về nguyên văn
    return text


def preprocess(
    text: str,
    segmenter: str = "pyvi",
    lowercase: bool = True,
) -> str:
    """
    Pipeline tiền xử lý đầy đủ: clean → lowercase → segment.
    """
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    text = segment_words(text, segmenter)
    return text


# ---------------------------------------------------------------------------
# Document data model
# ---------------------------------------------------------------------------

class LegalDocument:
    """
    Biểu diễn một văn bản pháp lý từ context_*.json.
    """

    __slots__ = ("doc_id", "name", "link", "passage", "source_file")

    def __init__(
        self,
        doc_id: int,
        name: str,
        link: str,
        passage: str,
        source_file: str = "",
    ) -> None:
        self.doc_id      = doc_id
        self.name        = name
        self.link        = link
        self.passage     = passage
        self.source_file = source_file

    def __repr__(self) -> str:
        return f"<LegalDocument id={self.doc_id} name='{self.name[:60]}...'>"


# ---------------------------------------------------------------------------
# Worker function for parallel file loading
# ---------------------------------------------------------------------------

def _parse_and_preprocess_file(args: Tuple[str, str, bool, float]) -> List[Tuple]:
    """
    Worker xử lý 1 file context_*.json độc lập trong process pool.
    """
    json_path, segmenter, lowercase, name_boost = args
    results = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return results

    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return results

    source_file = os.path.basename(json_path)

    for record in records:
        try:
            doc_id  = int(record["id"])
            name    = str(record.get("name",    ""))
            link    = str(record.get("link",    ""))
            passage = str(record.get("passage", ""))
        except (KeyError, ValueError):
            continue

        doc = LegalDocument(
            doc_id=doc_id,
            name=name,
            link=link,
            passage=passage,
            source_file=source_file,
        )

        raw_text = clean_text(name + " " + passage)

        processed_name    = preprocess(name,    segmenter, lowercase)
        processed_passage = preprocess(passage, segmenter, lowercase)

        boost_count  = max(1, int(name_boost))
        boosted_name = " ".join([processed_name] * boost_count)
        bm25_text    = boosted_name + " " + processed_passage

        results.append((doc, raw_text, bm25_text))

    return results


# ---------------------------------------------------------------------------
# Context corpus loader with Cache & Multiprocessing
# ---------------------------------------------------------------------------

def load_contexts(
    contexts_dir: str,
    segmenter: str = "pyvi",
    lowercase: bool = True,
    name_boost: float = 2.0,
    show_progress: bool = True,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    num_workers: Optional[int] = None,
) -> Tuple[List[LegalDocument], List[str], List[str]]:
    """
    Đọc toàn bộ file context_*.json từ thư mục `contexts_dir`.
    - Hỗ trợ đọc từ Cache (.cache/preprocessed_corpus.pkl) siêu tốc (<0.5s)
    - Hỗ trợ xử lý song song (Multiprocessing) trên toàn bộ nhân CPU

    Returns:
        docs       : Danh sách LegalDocument
        raw_texts  : Văn bản gốc cho Dense encoder
        bm25_texts : Văn bản đã tiền xử lý + boost tên cho BM25
    """
    # ── 1. Thử load từ cache đĩa ──────────────────────────────────────────
    if not force_rebuild and cache_path and os.path.isfile(cache_path):
        logger.info(f"Đang đọc preprocessed corpus từ CACHE đĩa: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            docs, raw_texts, bm25_texts = cached_data
            logger.info(f"🚀 Loaded {len(docs)} documents từ CACHE đĩa cực nhanh (<0.5s)!")
            return docs, raw_texts, bm25_texts
        except Exception as e:
            logger.warning(f"Lỗi khi đọc cache ({e}). Tiến hành load lại...")

    # ── 2. Đọc file từ thư mục ─────────────────────────────────────────────
    if not os.path.isdir(contexts_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục contexts: {contexts_dir}\n"
            "Hãy giải nén selected-contexts.zip vào thư mục data/"
        )

    json_files = sorted(glob.glob(os.path.join(contexts_dir, "context_*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"Không tìm thấy file context_*.json nào trong: {contexts_dir}"
        )

    cpu_count = os.cpu_count() or 4
    workers = num_workers or min(cpu_count, 8)
    logger.info(
        f"Tìm thấy {len(json_files)} file context. "
        f"Xử lý song song với {workers} CPU workers (segmenter='{segmenter}')..."
    )

    tasks = [
        (filepath, segmenter, lowercase, name_boost)
        for filepath in json_files
    ]

    docs: List[LegalDocument] = []
    raw_texts: List[str]      = []
    bm25_texts: List[str]     = []

    # Multiprocessing Parallel Execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parse_and_preprocess_file, task) for task in tasks]
        
        iterable = tqdm(as_completed(futures), total=len(futures), desc="Processing contexts", unit="file") \
                   if show_progress else as_completed(futures)
        
        for future in iterable:
            file_results = future.result()
            for doc, raw_text, bm25_text in file_results:
                docs.append(doc)
                raw_texts.append(raw_text)
                bm25_texts.append(bm25_text)

    # Sắp xếp lại theo doc_id để đảm bảo thứ tự nhất quán
    sorted_triplets = sorted(zip(docs, raw_texts, bm25_texts), key=lambda x: x[0].doc_id)
    docs       = [t[0] for t in sorted_triplets]
    raw_texts  = [t[1] for t in sorted_triplets]
    bm25_texts = [t[2] for t in sorted_triplets]

    logger.info(f"Loaded & processed {len(docs)} documents từ corpus.")

    # ── 3. Lưu cache đĩa để các lần chạy sau không phải tính lại ───────────
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((docs, raw_texts, bm25_texts), f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Đã lưu preprocessed corpus cache tại: {cache_path}")
        except Exception as e:
            logger.warning(f"Không thể lưu cache: {e}")

    return docs, raw_texts, bm25_texts


# ---------------------------------------------------------------------------
# Query loaders
# ---------------------------------------------------------------------------

def load_queries(file_path: str) -> List[Dict]:
    """
    Load file train.json hoặc public-official.json.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: List[Dict] = []

    if isinstance(data, list):
        for item in data:
            if "id" not in item or "question" not in item:
                logger.warning(f"Bỏ qua record thiếu field: {item}")
                continue
            queries.append(item)

    elif isinstance(data, dict):
        for qid, val in data.items():
            if isinstance(val, dict):
                entry = {"id": qid, **val}
            else:
                entry = {"id": qid, "question": str(val)}
            queries.append(entry)

    logger.info(
        f"Loaded {len(queries)} queries từ '{os.path.basename(file_path)}'."
    )
    return queries
