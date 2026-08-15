# UIT Data Science Challenge 2026 — LegalIR System

Hệ thống truy xuất văn bản pháp lý tiếng Việt theo kiến trúc Hybrid Retrieval (BM25 + Dense).

---

## Cấu trúc thư mục

```
UIT_DATA_SCIENCE_CHALLENGE_2026/
├── data/
│   ├── train.json                # Tập train (có ground truth)
│   ├── public-official.json      # Tập test chính thức
│   └── selected-contexts/        # Giải nén từ selected-contexts.zip
│       ├── context_0001.json
│       ├── context_0002.json
│       └── ...
├── legal_ir/
│   ├── __init__.py
│   ├── config.py         # Tất cả hyperparameters và đường dẫn
│   ├── data_loader.py    # Multi-processing parallel load & corpus disk cache
│   ├── bm25_retriever.py # Stage 1: BM25 sparse retrieval
│   ├── dense_retriever.py# Stage 2: Dense re-ranking (Sentence-Transformer)
│   ├── hybrid_ranker.py  # Kết hợp BM25 + Dense + Evaluation metrics
│   ├── exporter.py       # Xuất submission.json + submission.zip
│   └── main.py           # Entry point (CLI)

> ⚡ **Đã tối ưu tốc độ x100:** 
> 1. Dùng `pyvi` tách từ tiếng Việt siêu nhanh (~1-2s cho 8,500 file).
> 2. Đọc & tiền xử lý file bằng **Multiprocessing** chạy song song trên tất cả nhân CPU.
> 3. Tự động lưu **Corpus Cache đĩa** (`.cache/preprocessed_corpus.pkl`), lần sau khởi động chỉ mất **<0.5s**!
├── tests/
│   └── test_pipeline.py  # Unit tests
├── output/               # Kết quả xuất ra
│   ├── submission.json
│   └── submission.zip
├── .cache/               # Cache BM25 index + embeddings
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `underthesea` yêu cầu cài thêm model:
> ```bash
> python -c "from underthesea import word_tokenize; word_tokenize('test')"
> ```

---

## Chuẩn bị dữ liệu

1. Đặt `train.json` và `public-official.json` vào thư mục `data/`
2. Giải nén `selected-contexts.zip` vào `data/selected-contexts/`

```
data/
├── train.json
├── public-official.json
└── selected-contexts/
    ├── context_0001.json
    └── ...
```

---

## Cách chạy

### Tạo file nộp bài (submission.zip)

```bash
python -m legal_ir.main --mode predict
```

### Đánh giá trên tập train

```bash
python -m legal_ir.main --mode eval
```

### Tìm hyperparameter tốt nhất (Grid Search)

```bash
python -m legal_ir.main --mode tune
```

### Các tùy chọn thêm

```bash
# Chỉ dùng BM25 (không cần GPU, nhanh hơn)
python -m legal_ir.main --mode predict --no-dense

# Rebuild index từ đầu (bỏ qua cache)
python -m legal_ir.main --mode predict --force-rebuild

# Override alpha (trọng số BM25)
python -m legal_ir.main --mode eval --alpha 0.3

# Override threshold lọc document
python -m legal_ir.main --mode eval --threshold 0.15
```

### Chạy unit tests

```bash
python -m pytest tests/ -v
# hoặc
python tests/test_pipeline.py
```

---

## Kiến trúc Pipeline

```
Câu hỏi (tiếng Việt)
        │
        ▼
┌─────────────────────┐
│  Text Preprocessing │  ← Unicode normalize, clean, word segment (underthesea)
└─────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
BM25 Index  Dense Encoder
(rank_bm25) (Vietnamese Bi-Encoder)
   │         │
   │  Top-50 │  Top-20 re-rank
   └────┬────┘
        │
        ▼
┌─────────────────────┐
│  Hybrid Score Fusion │  ← α·BM25 + β·Dense
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Post-processing     │  ← Threshold filter, Top-5 clip
└─────────────────────┘
        │
        ▼
  submission.zip
```

---

## Cấu hình (config.py)

Chỉnh sửa `legal_ir/config.py` để điều chỉnh:

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `bm25.variant` | `BM25Okapi` | Biến thể BM25 (Okapi/L/Plus) |
| `bm25.top_k_stage1` | `50` | Số candidate lấy từ BM25 |
| `bm25.name_boost` | `2.0` | Hệ số tăng trọng tên văn bản |
| `dense.model_name` | `bkai-foundation-models/vietnamese-bi-encoder` | Model Sentence-Transformer |
| `dense.enabled` | `True` | Bật/tắt Dense retriever |
| `hybrid.alpha` | `0.4` | Trọng số BM25 |
| `hybrid.beta` | `0.6` | Trọng số Dense |
| `hybrid.score_threshold` | `0.10` | Ngưỡng tối thiểu để lấy doc |
| `hybrid.max_docs_per_query` | `5` | Giới hạn IDs/câu hỏi (luật thi) |
| `text.word_segmenter` | `underthesea` | Backend tách từ |

---

## Định dạng Output

`submission.json`:
```json
{
  "Q001": { "answer": ["12345", "67890"] },
  "Q002": { "answer": ["11111"] },
  "Q003": { "answer": [] }
}
```

> ⚠️ **Ràng buộc cuộc thi:** Mỗi câu hỏi **TỐI ĐA 5** `doc_id`. Vi phạm → điểm = 0.

---

## Độ đo đánh giá

- **Recall@5** (độ đo chính): Tỉ lệ văn bản đúng được tìm thấy trong Top-5
- **Precision@5** (độ đo phụ): Tỉ lệ văn bản đúng trong số 5 văn bản trả về
- **F1@5**: Trung bình điều hoà Recall@5 và Precision@5
