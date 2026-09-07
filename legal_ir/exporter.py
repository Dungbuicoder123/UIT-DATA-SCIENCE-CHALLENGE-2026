"""
LegalIR — Exporter
"""
import json, os, zipfile
def export_submission(query_ids, predictions, output_dir, max_docs=5):
    os.makedirs(output_dir, exist_ok=True)
    clipped = {qid: ans[:max_docs] for qid, ans in predictions.items()}
    submission = {str(qid): {"answer": [str(did) for did in clipped.get(qid, [])]} for qid in query_ids}
    
    json_path = os.path.join(output_dir, "submission.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
        
    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    print(f"✅ Đã xuất file thành công: {zip_path}")
    return zip_path
