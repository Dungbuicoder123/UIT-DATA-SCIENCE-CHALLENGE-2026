"""
=============================================================================
LegalIR — Submission Exporter
=============================================================================
• Đóng gói kết quả ra file submission.json đúng cấu trúc cuộc thi
• Nén thành submission.zip
• Validate kết quả trước khi xuất (kiểm tra ràng buộc ≤ 5 IDs)
=============================================================================
"""

import json
import logging
import os
import zipfile
from typing import Dict, List

logger = logging.getLogger(__name__)


def validate_predictions(
    predictions: Dict[str, List[str]],
    max_docs: int = 5,
) -> None:
    """
    Kiểm tra ràng buộc của cuộc thi trước khi xuất file:
    - Mỗi question_id không được trả về > max_docs IDs
    - Các doc_id phải là string

    Raises:
        ValueError nếu có vi phạm nghiêm trọng.
    """
    violations = []
    for qid, answers in predictions.items():
        if len(answers) > max_docs:
            violations.append(
                f"  • Query '{qid}': {len(answers)} IDs > {max_docs}"
            )
        for did in answers:
            if not isinstance(did, str):
                violations.append(
                    f"  • Query '{qid}': doc_id '{did}' không phải string!"
                )

    if violations:
        msg = (
            f"Phát hiện {len(violations)} vi phạm ràng buộc cuộc thi:\n"
            + "\n".join(violations)
            + "\n⚠ Các câu hỏi vi phạm sẽ bị gán điểm = 0!"
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        f"Validation OK: {len(predictions)} queries, "
        f"tất cả đều ≤ {max_docs} IDs/query."
    )


def build_submission_dict(
    query_ids: List[str],
    predictions: Dict[str, List[str]],
) -> Dict:
    """
    Tạo dict đúng định dạng cuộc thi:
    {
        "Q001": { "answer": ["doc_id_A", "doc_id_B"] },
        "Q002": { "answer": ["doc_id_C"] },
        ...
    }

    Args:
        query_ids   : Danh sách tất cả question_id (kể cả không có kết quả)
        predictions : Dict {question_id → [doc_ids]}
    """
    submission = {}
    for qid in query_ids:
        answers = predictions.get(qid, [])
        submission[str(qid)] = {"answer": [str(did) for did in answers]}
    return submission


def export_submission(
    query_ids: List[str],
    predictions: Dict[str, List[str]],
    output_dir: str,
    json_filename: str = "submission.json",
    zip_filename: str = "submission.zip",
    max_docs: int = 5,
    skip_validation: bool = False,
) -> str:
    """
    Xuất kết quả ra file JSON và nén thành ZIP.

    Args:
        query_ids       : Danh sách question_id của test set
        predictions     : Dict {question_id → List[doc_id_str]}
        output_dir      : Thư mục xuất file
        json_filename   : Tên file JSON (mặc định "submission.json")
        zip_filename    : Tên file ZIP (mặc định "submission.zip")
        max_docs        : Giới hạn tối đa IDs/câu hỏi
        skip_validation : Bỏ qua bước validation (không khuyến nghị)

    Returns:
        Đường dẫn tuyệt đối đến file ZIP đã tạo.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clip predictions tối đa max_docs để đảm bảo an toàn
    clipped = {qid: ans[:max_docs] for qid, ans in predictions.items()}

    # Validate
    if not skip_validation:
        validate_predictions(clipped, max_docs)

    # Build submission dict
    submission = build_submission_dict(query_ids, clipped)

    # Xuất JSON
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã xuất submission JSON: {json_path}")

    # Nén thành ZIP
    zip_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Thêm file JSON vào ZIP với tên đúng yêu cầu
        zf.write(json_path, arcname=json_filename)

    # Thống kê
    zip_size_kb = os.path.getsize(zip_path) / 1024
    total_answers = sum(len(v["answer"]) for v in submission.values())
    avg_answers   = total_answers / len(submission) if submission else 0

    logger.info(
        f"Đã nén thành: {zip_path} ({zip_size_kb:.1f} KB)\n"
        f"  Tổng số queries : {len(submission)}\n"
        f"  Tổng số answers : {total_answers}\n"
        f"  Trung bình IDs/query: {avg_answers:.2f}"
    )

    print(f"\n✅ Submission đã sẵn sàng!")
    print(f"   📄 JSON : {json_path}")
    print(f"   🗜  ZIP  : {zip_path} ({zip_size_kb:.1f} KB)")
    print(f"   📊 Avg IDs/query: {avg_answers:.2f}\n")

    return zip_path
