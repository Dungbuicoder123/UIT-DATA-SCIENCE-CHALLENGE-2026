"""
=============================================================================
LegalIR — Data Loader & Text Preprocessor
=============================================================================
Handles:
  • Loading and parsing context JSON files from selected-contexts/
  • Loading train.json and public-official.json (queries)
  • Vietnamese text cleaning, Unicode normalisation, word segmentation

Ban da sua (so voi ban goc):
  1. segment_words(): chia nho van ban dai thanh cac chunk truoc khi dua vao
     underthesea/pyvi, tranh treo/cham bat thuong voi cac file cuc dai (vd
     QCVN/TCVN co the len toi 4+ trieu ky tu) - day la nguyen nhan chinh khien
     tien trinh bi ngat giua chung luc chay tren Colab.
  2. load_contexts(): them checkpoint (luu tam moi 500 file) de neu bi ngat
     giua chung (Colab disconnect, het RAM, ...) thi lan chay lai se tu dong
     resume tu cho da dung, khong phai load lai tu dau.
=============================================================================
"""

import glob
import json
import logging
import os
import pickle
import re
import unicodedata
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


def segment_words(text: str, segmenter: str = "underthesea", max_chunk_chars: int = 200000) -> str:
    """
    Tách từ tiếng Việt.
    - "underthesea" : dùng thư viện underthesea (chính xác hơn)
    - "pyvi"        : dùng thư viện pyvi (nhanh hơn)
    - "none"        : không tách từ (dùng khi debug nhanh)

    Trả về chuỗi đã tách từ (từ ghép được nối bằng dấu gạch dưới: _).

    LUU Y (sua so voi ban goc): voi van ban dai hon `max_chunk_chars`, ham
    se chia nho theo dong roi tokenize tung chunk mot thay vi dua nguyen ca
    van ban (co the toi vai trieu ky tu) vao thang mot lan - do la nguyen
    nhan gay treo/cham voi cac file cuc dai (QCVN, TCVN...).
    """
    if not text:
        return ""

    def _tokenize_chunk(chunk: str, seg: str) -> str:
        if seg == "underthesea":
            try:
                from underthesea import word_tokenize  # type: ignore
                return word_tokenize(chunk, format="text")
            except ImportError:
                logger.warning("underthesea không được cài đặt. Chuyển sang pyvi.")
                return _tokenize_chunk(chunk, "pyvi")

        if seg == "pyvi":
            try:
                from pyvi import ViTokenizer  # type: ignore
                return ViTokenizer.tokenize(chunk)
            except ImportError:
                logger.warning("pyvi không được cài đặt. Không tách từ.")
                return chunk

        # "none" — trả về nguyên văn
        return chunk

    if len(text) <= max_chunk_chars:
        return _tokenize_chunk(text, segmenter)

    # Van ban qua dai -> chia thanh cac chunk <= max_chunk_chars, gop theo
    # dong de khong cat dut giua tu. Tokenize tung chunk roi noi lai bang
    # khoang trang.
    chunks: List[str] = []
    buf = ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > max_chunk_chars:
            chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)

    return " ".join(_tokenize_chunk(c, segmenter) for c in chunks)


def preprocess(
    text: str,
    segmenter: str = "underthesea",
    lowercase: bool = True,
) -> str:
    """
    Pipeline tiền xử lý đầy đủ: clean → lowercase → segment.
    Đây là hàm duy nhất được gọi ở nơi khác.
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

    Attributes:
        doc_id   : ID của văn bản (int, được chuyển sang str khi xuất)
        name     : Tên/tiêu đề văn bản
        link     : Đường dẫn nguồn
        passage  : Nội dung văn bản đầy đủ
        source_file: File JSON gốc (để debug)
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
# Context corpus loader
# ---------------------------------------------------------------------------

def load_contexts(
    contexts_dir: str,
    segmenter: str = "underthesea",
    lowercase: bool = True,
    name_boost: float = 2.0,
    show_progress: bool = True,
    checkpoint_path: str = "loader_checkpoint.pkl",
    checkpoint_every: int = 500,
) -> Tuple[List[LegalDocument], List[str], List[str]]:
    """
    Đọc toàn bộ file context_*.json từ thư mục `contexts_dir`.

    Returns:
        docs       : Danh sách LegalDocument (thứ tự ổn định)
        raw_texts  : Văn bản gốc (passage) — dùng cho dense encoder
        bm25_texts : Văn bản đã tiền xử lý + boost tên — dùng cho BM25

    Mỗi entry trong bm25_texts = [name (lặp name_boost lần)] + passage
    để tăng trọng số cho tiêu đề văn bản.

    LUU Y (sua so voi ban goc): co checkpoint - cu moi `checkpoint_every`
    file xu ly xong se luu tam ra `checkpoint_path`. Neu tien trinh bi ngat
    giua chung (Colab disconnect, het RAM,...), goi lai ham nay voi cung
    `checkpoint_path` se tu dong resume tu cho da dung thay vi chay lai tu
    dau. Muon chay lai tu dau hoan toan thi xoa file checkpoint truoc.
    """
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

    logger.info(f"Tìm thấy {len(json_files)} file context.")

    docs: List[LegalDocument]  = []
    raw_texts: List[str]       = []
    bm25_texts: List[str]      = []
    start_idx = 0

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                docs, raw_texts, bm25_texts, start_idx = pickle.load(f)
            logger.info(
                f"Tìm thấy checkpoint: đã xử lý {start_idx}/{len(json_files)} file trước đó. "
                "Tiếp tục từ vị trí này (xoá file checkpoint nếu muốn chạy lại từ đầu)."
            )
        except Exception as e:
            logger.warning(f"Không đọc được checkpoint ({e}), chạy lại từ đầu.")
            docs, raw_texts, bm25_texts, start_idx = [], [], [], 0

    iterable = tqdm(
        enumerate(json_files),
        total=len(json_files),
        initial=start_idx,
        desc="Loading contexts",
        unit="file",
    ) if show_progress else enumerate(json_files)

    for i, json_path in iterable:
        if i < start_idx:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Mỗi file context_*.json có thể chứa 1 dict hoặc 1 list of dict
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            logger.warning(f"Bỏ qua file có định dạng lạ: {json_path}")
            continue

        for record in records:
            try:
                doc_id  = int(record["id"])
                name    = str(record.get("name",    ""))
                link    = str(record.get("link",    ""))
                passage = str(record.get("passage", ""))
            except (KeyError, ValueError) as e:
                logger.warning(f"Bỏ qua record lỗi trong {json_path}: {e}")
                continue

            doc = LegalDocument(
                doc_id=doc_id,
                name=name,
                link=link,
                passage=passage,
                source_file=os.path.basename(json_path),
            )
            docs.append(doc)

            # Văn bản gốc cho dense encoder (giữ dấu tiếng Việt)
            raw_text = clean_text(name + " " + passage)
            raw_texts.append(raw_text)

            # Văn bản đã xử lý cho BM25
            processed_name    = preprocess(name,    segmenter, lowercase)
            processed_passage = preprocess(passage, segmenter, lowercase)

            # Boost tên tài liệu: lặp lại `name_boost` lần → tăng trọng số
            boost_count  = max(1, int(name_boost))
            boosted_name = " ".join([processed_name] * boost_count)
            bm25_text    = boosted_name + " " + processed_passage
            bm25_texts.append(bm25_text)

        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump((docs, raw_texts, bm25_texts, i + 1), f)
            logger.info(f"Đã lưu checkpoint tại {i + 1}/{len(json_files)} file.")

    # Xu ly xong toan bo -> xoa checkpoint (khong con can resume nua)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logger.info(f"Loaded {len(docs)} documents từ corpus.")
    return docs, raw_texts, bm25_texts


# ---------------------------------------------------------------------------
# Query loaders
# ---------------------------------------------------------------------------

def load_queries(file_path: str) -> List[Dict]:
    """
    Load file train.json hoặc public-official.json.

    Expected format (list of objects):
      [{"id": "Q001", "question": "Điều kiện để..."}, ...]
    Hoặc dict keyed by question ID:
      {"Q001": {"question": "..."}, ...}

    Returns:
        list of dict với keys: "id", "question"
        (và "answer" nếu là train set)
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
                # val là chuỗi câu hỏi
                entry = {"id": qid, "question": str(val)}
            queries.append(entry)

    logger.info(
        f"Loaded {len(queries)} queries từ '{os.path.basename(file_path)}'."
    )
    return queries