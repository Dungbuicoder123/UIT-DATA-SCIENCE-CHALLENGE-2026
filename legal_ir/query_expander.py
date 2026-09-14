"""
LegalIR — Query Expander (Nâng cấp: Trích xuất thực thể điều luật & Bung viết tắt pháp lý)
"""
import re
from typing import Dict, List

# Từ điển viết tắt pháp lý tiếng Việt phổ biến
LEGAL_ABBREVIATIONS: Dict[str, str] = {
    # Loại văn bản
    r"\bNĐ\b": "Nghị định",
    r"\bNĐ-CP\b": "Nghị định Chính phủ",
    r"\bTT\b": "Thông tư",
    r"\bTTLT\b": "Thông tư liên tịch",
    r"\bQĐ\b": "Quyết định",
    r"\bQĐ-TTg\b": "Quyết định Thủ tướng Chính phủ",
    r"\bQĐ-BTC\b": "Quyết định Bộ Tài chính",
    r"\bQĐ-BGDĐT\b": "Quyết định Bộ Giáo dục và Đào tạo",
    r"\bCT\b": "Chỉ thị",
    r"\bNQ\b": "Nghị quyết",
    r"\bNQ-CP\b": "Nghị quyết Chính phủ",
    r"\bPL\b": "Pháp lệnh",
    r"\bQCVN\b": "Quy chuẩn kỹ thuật quốc gia",
    r"\bTCVN\b": "Tiêu chuẩn Việt Nam",
    r"\bCV\b": "Công văn",
    # Bộ luật
    r"\bBLDS\b": "Bộ luật Dân sự",
    r"\bBLHS\b": "Bộ luật Hình sự",
    r"\bBLTTDS\b": "Bộ luật Tố tụng dân sự",
    r"\bBLTTHS\b": "Bộ luật Tố tụng hình sự",
    r"\bBLLĐ\b": "Bộ luật Lao động",
    # Tên luật phổ biến
    r"\bLuật\s+GTĐB\b": "Luật Giao thông đường bộ",
    r"\bLuật\s+SHTT\b": "Luật Sở hữu trí tuệ",
    r"\bLuật\s+DN\b": "Luật Doanh nghiệp",
    r"\bLuật\s+ĐĐ\b": "Luật Đất đai",
    r"\bLuật\s+XD\b": "Luật Xây dựng",
    r"\bLuật\s+BHXH\b": "Luật Bảo hiểm xã hội",
    r"\bLuật\s+BVMT\b": "Luật Bảo vệ môi trường",
    r"\bLuật\s+TTTM\b": "Luật Thương mại",
    r"\bLuật\s+HN&GĐ\b": "Luật Hôn nhân và Gia đình",
    # Bộ ngành và cơ quan
    r"\bBTC\b": "Bộ Tài chính",
    r"\bBCA\b": "Bộ Công an",
    r"\bBQP\b": "Bộ Quốc phòng",
    r"\bBNNPTNT\b": "Bộ Nông nghiệp và Phát triển nông thôn",
    r"\bBYT\b": "Bộ Y tế",
    r"\bBGDĐT\b": "Bộ Giáo dục và Đào tạo",
    r"\bBCT\b": "Bộ Công Thương",
    r"\bBKHĐT\b": "Bộ Kế hoạch và Đầu tư",
    r"\bBTTTT\b": "Bộ Thông tin và Truyền thông",
    r"\bBXD\b": "Bộ Xây dựng",
    r"\bBLĐTBXH\b": "Bộ Lao động Thương binh và Xã hội",
    r"\bBTP\b": "Bộ Tư pháp",
    r"\bBGTVT\b": "Bộ Giao thông Vận tải",
    r"\bBNV\b": "Bộ Nội vụ",
    r"\bBTNMT\b": "Bộ Tài nguyên và Môi trường",
    r"\bBVHTTDL\b": "Bộ Văn hóa Thể thao và Du lịch",
    r"\bTTg\b": "Thủ tướng Chính phủ",
    r"\bUBND\b": "Ủy ban nhân dân",
    r"\bHĐND\b": "Hội đồng nhân dân",
    r"\bNHNN\b": "Ngân hàng Nhà nước",
}

# Compile sẵn các pattern viết tắt
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.UNICODE), replacement)
    for pattern, replacement in LEGAL_ABBREVIATIONS.items()
]

# Regex nhận diện định danh Điều khoản pháp luật trong câu hỏi (VD: "Điều 15", "Khoản 2 Điều 10")
_ARTICLE_REF_RE = re.compile(r'\b(Điều\s+\d+[\s\d,\–\-]*(?:\s+khoản\s+\d+)?)\b', re.IGNORECASE | re.UNICODE)


def expand_legal_abbreviations(text: str) -> str:
    """Bung từ viết tắt pháp lý trong văn bản."""
    if not text:
        return text
    for pattern, replacement in _COMPILED_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def extract_and_boost_articles(query: str) -> str:
    """
    Trích xuất các thực thể Điều luật có trong câu hỏi và lặp lại chúng (boosting)
    để trọng số khi tính toán Lexical/BM25 hoặc Dense tập trung mạnh vào đúng điều luật đó.
    """
    matches = _ARTICLE_REF_RE.findall(query)
    if not matches:
        return query
    
    # Nhân đôi các từ khóa điều luật tìm thấy để tăng độ ưu tiên khớp (Query Enrichment / Boosting)
    boosted_part = " ".join(matches * 2)
    return f"{query} {boosted_part}"


def expand_query(query: str) -> str:
    """
    Pipeline tiền xử lý nâng cao cho câu hỏi pháp lý:
    1. Bung từ viết tắt (NĐ -> Nghị định, BLDS -> Bộ luật Dân sự...)
    2. Trích xuất và nhân bản định danh điều luật (Boosting)
    3. Chuẩn hóa khoảng trắng.
    """
    if not query:
        return ""
    
    # Bước 1: Bung viết tắt
    expanded = expand_legal_abbreviations(query)
    
    # Bước 2: Trích xuất và boost điều luật
    expanded = extract_and_boost_articles(expanded)
    
    # Bước 3: Chuẩn hóa khoảng trắng
    expanded = re.sub(r'\s+', ' ', expanded).strip()
    return expanded


if __name__ == "__main__":
    # Demo kiểm tra nhanh
    samples = [
        "Theo Điều 24 Nghị định 100/2019/NĐ-CP xử phạt thế nào?",
        "Vi phạm khoản 2 Điều 15 Luật Đất đai bị thu hồi không?",
    ]
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {expand_query(s)}")
        print()