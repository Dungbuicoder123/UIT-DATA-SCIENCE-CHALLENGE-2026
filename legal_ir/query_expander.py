"""
LegalIR — Query Expander (Bung từ viết tắt pháp lý tiếng Việt)
"""
import re
from typing import Dict

# Từ điển viết tắt pháp lý tiếng Việt phổ biến
# Key: pattern (case-insensitive), Value: dạng đầy đủ
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
    # Bộ ngành
    r"\bBTC\b": "Bộ Tài chính",
    r"\bBCA\b": "Bộ Công an",
    r"\bBQP\b": "Bộ Quốc phòng",
    r"\bBNN\b": "Bộ Nông nghiệp",
    r"\bBNNPTNT\b": "Bộ Nông nghiệp và Phát triển nông thôn",
    r"\bBYT\b": "Bộ Y tế",
    r"\bBGD&ĐT\b": "Bộ Giáo dục và Đào tạo",
    r"\bBGDĐT\b": "Bộ Giáo dục và Đào tạo",
    r"\bBCT\b": "Bộ Công Thương",
    r"\bBKH&ĐT\b": "Bộ Kế hoạch và Đầu tư",
    r"\bBKHĐT\b": "Bộ Kế hoạch và Đầu tư",
    r"\bBTT&TT\b": "Bộ Thông tin và Truyền thông",
    r"\bBTTTT\b": "Bộ Thông tin và Truyền thông",
    r"\bBXD\b": "Bộ Xây dựng",
    r"\bBLĐTB&XH\b": "Bộ Lao động Thương binh và Xã hội",
    r"\bBLĐTBXH\b": "Bộ Lao động Thương binh và Xã hội",
    r"\bBTP\b": "Bộ Tư pháp",
    r"\bBGT\b": "Bộ Giao thông",
    r"\bBGTVT\b": "Bộ Giao thông Vận tải",
    r"\bBNV\b": "Bộ Nội vụ",
    r"\bBNG\b": "Bộ Ngoại giao",
    r"\bBNLĐ\b": "Bộ Nhà nước",
    r"\bBKH&CN\b": "Bộ Khoa học và Công nghệ",
    r"\bBKHCN\b": "Bộ Khoa học và Công nghệ",
    r"\bBTN&MT\b": "Bộ Tài nguyên và Môi trường",
    r"\bBTNMT\b": "Bộ Tài nguyên và Môi trường",
    r"\bBVHTT&DL\b": "Bộ Văn hóa Thể thao và Du lịch",
    r"\bBVHTTDL\b": "Bộ Văn hóa Thể thao và Du lịch",
    # Cơ quan khác
    r"\bTTg\b": "Thủ tướng Chính phủ",
    r"\bCTTg\b": "Chủ tịch nước",
    r"\bVKSND\b": "Viện Kiểm sát nhân dân",
    r"\bTAND\b": "Tòa án nhân dân",
    r"\bUBND\b": "Ủy ban nhân dân",
    r"\bHĐND\b": "Hội đồng nhân dân",
    r"\bHĐTP\b": "Hội đồng Thẩm phán",
    r"\bNHNN\b": "Ngân hàng Nhà nước",
    r"\bBHXH\b": "Bảo hiểm xã hội",
    r"\bBHYT\b": "Bảo hiểm y tế",
    # Khái niệm pháp lý
    r"\bKTTM\b": "kinh tế thị trường",
    r"\bKKT\b": "khu kinh tế",
    r"\bKCN\b": "khu công nghiệp",
    r"\bKCX\b": "khu chế xuất",
    r"\bKKTCK\b": "khu kinh tế cửa khẩu",
    r"\bGP\b": "giấy phép",
    r"\bGCN\b": "giấy chứng nhận",
    r"\bHĐLĐ\b": "hợp đồng lao động",
    r"\bHĐMB\b": "hợp đồng mua bán",
    r"\bHĐKT\b": "hợp đồng kinh tế",
    r"\bXPVP\b": "xử phạt vi phạm",
    r"\bXPHC\b": "xử phạt hành chính",
    r"\bTNHH\b": "trách nhiệm hữu hạn",
    r"\bCTCP\b": "công ty cổ phần",
    r"\bDNNN\b": "doanh nghiệp nhà nước",
    r"\bDNTN\b": "doanh nghiệp tư nhân",
    r"\bTHCS\b": "trung học cơ sở",
    r"\bTHPT\b": "trung học phổ thông",
}

# Compile all patterns once for performance
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.UNICODE), replacement)
    for pattern, replacement in LEGAL_ABBREVIATIONS.items()
]


def expand_legal_abbreviations(text: str) -> str:
    """
    Bung từ viết tắt pháp lý trong văn bản.
    Ví dụ: "NĐ 100/2019/NĐ-CP" → "Nghị định 100/2019/Nghị định Chính phủ"
    """
    if not text:
        return text
    for pattern, replacement in _COMPILED_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def expand_query(query: str) -> str:
    """
    Tiền xử lý câu hỏi pháp lý: bung viết tắt và chuẩn hoá khoảng trắng.
    """
    expanded = expand_legal_abbreviations(query)
    # Chuẩn hoá khoảng trắng thừa
    expanded = re.sub(r'\s+', ' ', expanded).strip()
    return expanded


if __name__ == "__main__":
    # Demo test
    samples = [
        "NĐ 100/2019/NĐ-CP quy định gì về BHXH?",
        "TT 01/2020/TT-BNNPTNT hướng dẫn thực hiện BLDS không?",
        "UBND tỉnh có được ban hành QĐ về KCN không?",
        "Theo BLHS 2015, tội trốn thuế bị xử lý như thế nào?",
        "Luật GTĐB quy định tốc độ tối đa trong KCN là bao nhiêu?",
    ]
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {expand_query(s)}")
        print()
