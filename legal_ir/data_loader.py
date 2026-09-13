"""
LegalIR — Data Loader
Nâng cấp: Article-Aware Chunking + Document Header Injection + Query Expansion
"""
import glob, json, logging, os, pickle, re, unicodedata
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex nhận dạng ranh giới Điều luật trong văn bản pháp luật tiếng Việt
# Khớp: "Điều 1.", "Điều 12:", "Điều 3 .", "I.", "II.", "1.", "a)", ...
# ---------------------------------------------------------------------------
_ARTICLE_BOUNDARY_RE = re.compile(
    r'(?:^|\n)'                              # Đầu dòng
    r'(?:'
        r'Điều\s+\d+[\.\:\s]'               # "Điều 1." / "Điều 12:"
        r'|Chương\s+[IVXLC\d]+[\.\:\s]'    # "Chương I." / "Chương 2:"
        r'|Mục\s+\d+[\.\:\s]'              # "Mục 1."
        r'|[IVXLC]{1,5}\.\s'               # "I. ", "II. "
        r'|\d{1,3}\.\s+[A-ZÁÀẢÃẠĂẮẶẲẴ]'   # "1. Phạm vi áp dụng"
    r')',
    re.MULTILINE | re.UNICODE
)

# Regex trích xuất số hiệu văn bản từ passage (dòng đầu) / link
_DOC_NUMBER_RE = re.compile(
    r'(?:'
        r'\d{1,4}/\d{4}/[A-ZĐÀ\-]+'         # VD: 100/2019/NĐ-CP
        r'|\d{1,4}/[A-ZĐÀ\-]+-\w+'           # VD: 569/QĐ-TTg
        r'|QCVN\s+[\d\-\:\/\w]+'             # VD: QCVN 02-30:2018/BNNPTNT
        r'|TCVN\s+[\d\-\:\/\w]+'             # VD: TCVN 13268-1:2021
    r')',
    re.UNICODE
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def segment_words(text: str, segmenter: str = "underthesea") -> str:
    if not text:
        return ""
    if segmenter == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize(text)
        except Exception:
            return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text


def preprocess(text: str, segmenter: str = "underthesea", lowercase: bool = True) -> str:
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    return segment_words(text, segmenter)


# ---------------------------------------------------------------------------
# Tiện ích trích xuất Header văn bản
# ---------------------------------------------------------------------------

def _extract_doc_header_from_passage(passage: str) -> str:
    """Lấy tối đa 3 dòng đầu của passage làm tiêu đề văn bản."""
    lines = [ln.strip() for ln in passage.split("\n") if ln.strip()]
    return " — ".join(lines[:3]) if lines else ""


def _extract_doc_number(passage: str, link: str) -> str:
    """Trích xuất số hiệu văn bản từ nội dung hoặc URL."""
    # Thử trong 500 ký tự đầu của passage
    snippet = passage[:500]
    m = _DOC_NUMBER_RE.search(snippet)
    if m:
        return m.group(0)
    # Thử trong link
    m = _DOC_NUMBER_RE.search(link)
    if m:
        return m.group(0)
    return ""


def _extract_doc_type(passage: str) -> str:
    """Trích xuất loại văn bản (Nghị định / Thông tư / Luật / ...)."""
    snippet = passage[:300].upper()
    for kw in ["NGHỊ ĐỊNH", "THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ QUYẾT",
                "THÔNG TƯ LIÊN TỊCH", "PHÁP LỆNH", "CHỈ THỊ",
                "LUẬT", "BỘ LUẬT", "HIẾN PHÁP", "QUY CHUẨN", "TIÊU CHUẨN"]:
        if kw in snippet:
            return kw.title()
    return ""


def build_chunk_header(doc_type: str, doc_number: str, name_field: str) -> str:
    """
    Tạo chuỗi tiêu đề ngắn gọn, rõ ràng để inject vào đầu mỗi chunk.
    Ví dụ: "[Nghị định 100/2019/NĐ-CP]"
           "[Quyết định 569/QĐ-TTg - Quyet-dinh-569...]"
    """
    parts = []
    if doc_type and doc_number:
        parts.append(f"{doc_type} {doc_number}")
    elif doc_number:
        parts.append(doc_number)
    elif doc_type:
        parts.append(doc_type)
    if name_field and len(name_field) < 120:
        parts.append(name_field)
    if not parts:
        return ""
    return f"[{' — '.join(parts)}]"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text_simple(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Sliding-window chunking theo số từ (fallback khi không có cấu trúc Điều)."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def chunk_text_article_aware(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Article-Aware Chunking:
    1. Tách văn bản tại ranh giới Điều / Chương / Mục.
    2. Nếu một Điều quá dài (> chunk_size từ), áp dụng sliding-window bên trong.
    3. Nếu một Điều quá ngắn, gộp với Điều kế tiếp.
    """
    # Tìm tất cả ranh giới điều luật
    boundaries = [m.start() for m in _ARTICLE_BOUNDARY_RE.finditer(text)]

    # Nếu không có cấu trúc điều hoặc chỉ có 1 phần, dùng sliding window
    if len(boundaries) <= 1:
        return chunk_text_simple(text, chunk_size, chunk_overlap)

    # Tách thành các đoạn theo ranh giới điều
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

    # Phần header trước Điều 1 (nếu có)
    pre_header = text[: boundaries[0]].strip()

    chunks: List[str] = []
    current_words: List[str] = pre_header.split() if pre_header else []

    for seg in segments:
        seg_words = seg.split()

        # Nếu segment đơn lẻ đã vượt chunk_size → chia nhỏ bằng sliding window
        if len(seg_words) > chunk_size:
            # Đẩy buffer hiện tại vào chunks trước
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            sub_chunks = chunk_text_simple(seg, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue

        # Nếu thêm segment vào buffer mà không vượt chunk_size → gộp
        if len(current_words) + len(seg_words) <= chunk_size:
            current_words.extend(seg_words)
        else:
            # Flush buffer
            if current_words:
                chunks.append(" ".join(current_words))
            # Bắt đầu buffer mới với overlap từ chunk trước
            overlap_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
            current_words = overlap_words + seg_words

    # Flush buffer cuối
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks if chunks else chunk_text_simple(text, chunk_size, chunk_overlap)


def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, article_aware: bool = True
) -> List[str]:
    """Điểm vào hợp nhất: chọn chiến lược chunking dựa trên cấu hình."""
    if article_aware:
        return chunk_text_article_aware(text, chunk_size, chunk_overlap)
    return chunk_text_simple(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Load Corpus
# ---------------------------------------------------------------------------

def load_contexts(
    contexts_dir: str,
    segmenter: str = "underthesea",
    lowercase: bool = True,
    name_boost: float = 2.0,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    do_chunking: bool = True,
    article_aware: bool = True,
    inject_header: bool = True,
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """
    Load và chunk toàn bộ corpus.

    Trả về:
        chunk_ids       : ID của từng chunk, dạng "{doc_id}_{c_idx}"
        raw_chunk_texts : Text thô (dùng cho Dense embedding & Cross-Encoder)
        bm25_chunk_texts: Text đã tiền xử lý (dùng cho BM25)
        chunk_to_doc_map: {chunk_id -> doc_id}
    """
    json_files = sorted(glob.glob(os.path.join(contexts_dir, "context_*.json")))
    chunk_ids: List[str] = []
    raw_chunk_texts: List[str] = []
    bm25_chunk_texts: List[str] = []
    chunk_to_doc_map: Dict[str, int] = {}

    for json_path in tqdm(json_files, desc="Loading & Chunking contexts"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [data] if isinstance(data, dict) else data

        for record in records:
            try:
                doc_id = int(record["id"])
                name_field = str(record.get("name", ""))
                passage = str(record.get("passage", ""))
                link = str(record.get("link", ""))
            except (KeyError, ValueError):
                continue

            # Trích xuất metadata từ văn bản
            doc_type = _extract_doc_type(passage)
            doc_number = _extract_doc_number(passage, link)
            header_str = build_chunk_header(doc_type, doc_number, name_field) if inject_header else ""

            # Chuẩn hoá passage
            passage_clean = clean_text(passage)

            # Chunking
            p_chunks = (
                chunk_text(passage_clean, chunk_size, chunk_overlap, article_aware)
                if do_chunking
                else [passage_clean]
            )

            # Chuẩn bị name boost cho BM25
            boost_count = max(1, int(name_boost))
            processed_name = preprocess(name_field, segmenter, lowercase) if name_field else ""
            boosted_name = " ".join([processed_name] * boost_count) if processed_name else ""
            # Nếu không có name field nhưng có header, boost header thay thế
            if not processed_name and header_str:
                processed_header = preprocess(header_str, segmenter, lowercase)
                boosted_name = " ".join([processed_header] * boost_count)

            for c_idx, chunk_str in enumerate(p_chunks):
                cid = f"{doc_id}_{c_idx}"
                chunk_ids.append(cid)
                chunk_to_doc_map[cid] = doc_id

                # Raw text: header + chunk (dùng cho Dense & Cross-Encoder)
                if header_str:
                    raw_text = f"{header_str} {chunk_str}"
                else:
                    raw_text = chunk_str
                raw_chunk_texts.append(raw_text)

                # BM25 text: boosted name + preprocessed chunk
                processed_chunk = preprocess(chunk_str, segmenter, lowercase)
                bm25_text = f"{boosted_name} {processed_chunk}".strip()
                bm25_chunk_texts.append(bm25_text)

    logger.info(
        f"Loaded {len(json_files)} docs → {len(chunk_ids)} chunks "
        f"(article_aware={article_aware}, inject_header={inject_header})"
    )
    return chunk_ids, raw_chunk_texts, bm25_chunk_texts, chunk_to_doc_map


# ---------------------------------------------------------------------------
# Load Queries
# ---------------------------------------------------------------------------

def load_queries(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    if isinstance(data, list):
        for item in data:
            if "id" in item and "question" in item:
                queries.append(item)
    elif isinstance(data, dict):
        for qid, val in data.items():
            if isinstance(val, dict):
                entry = {"id": qid, **val}
            else:
                entry = {"id": qid, "question": str(val)}
            queries.append(entry)
    return queries
