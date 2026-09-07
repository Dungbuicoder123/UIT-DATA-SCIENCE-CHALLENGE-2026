"""
LegalIR — Data Loader (Tích hợp Semantic Chunking)
"""
import glob, json, logging, os, pickle, re, unicodedata
from typing import Dict, List, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

def normalize_unicode(text: str) -> str: return unicodedata.normalize("NFC", text)
def clean_text(text: str) -> str:
    if not text: return ""
    text = normalize_unicode(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def segment_words(text: str, segmenter: str = "underthesea") -> str:
    if not text: return ""
    if segmenter == "pyvi":
        try: 
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize(text)
        except: return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except: return text

def preprocess(text: str, segmenter: str = "underthesea", lowercase: bool = True) -> str:
    text = clean_text(text)
    if lowercase: text = text.lower()
    return segment_words(text, segmenter)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Cắt văn bản thành các chunk (cửa sổ trượt) dựa trên số từ"""
    words = text.split()
    if len(words) <= chunk_size: return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def load_contexts(
    contexts_dir: str, segmenter: str = "underthesea", lowercase: bool = True, name_boost: float = 2.0,
    chunk_size: int = 256, chunk_overlap: int = 64, do_chunking: bool = True
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """Trả về: chunk_ids, raw_chunk_texts, bm25_chunk_texts, chunk_to_doc_map"""
    json_files = sorted(glob.glob(os.path.join(contexts_dir, "context_*.json")))
    chunk_ids, raw_chunk_texts, bm25_chunk_texts, chunk_to_doc_map = [], [], [], {}
    
    for json_path in tqdm(json_files, desc="Loading and Chunking contexts"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [data] if isinstance(data, dict) else data
        
        for record in records:
            try:
                doc_id = int(record["id"])
                name = str(record.get("name", ""))
                passage = str(record.get("passage", ""))
            except: continue
            
            boost_count = max(1, int(name_boost))
            processed_name = preprocess(name, segmenter, lowercase)
            boosted_name = " ".join([processed_name] * boost_count)
            raw_name = clean_text(name)

            passage_clean = clean_text(passage)
            p_chunks = chunk_text(passage_clean, chunk_size, chunk_overlap) if do_chunking else [passage_clean]
            
            for c_idx, chunk_str in enumerate(p_chunks):
                cid = f"{doc_id}_{c_idx}"  # ID của chunk (vd: 1234_0)
                chunk_ids.append(cid)
                chunk_to_doc_map[cid] = doc_id
                
                raw_chunk_texts.append(f"{raw_name} {chunk_str}")
                processed_chunk = preprocess(chunk_str, segmenter, lowercase)
                bm25_chunk_texts.append(f"{boosted_name} {processed_chunk}")
                
    return chunk_ids, raw_chunk_texts, bm25_chunk_texts, chunk_to_doc_map

def load_queries(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
    queries = []
    if isinstance(data, list):
        for item in data:
            if "id" in item and "question" in item: queries.append(item)
    elif isinstance(data, dict):
        for qid, val in data.items():
            entry = {"id": qid, **val} if isinstance(val, dict) else {"id": qid, "question": str(val)}
            queries.append(entry)
    return queries
