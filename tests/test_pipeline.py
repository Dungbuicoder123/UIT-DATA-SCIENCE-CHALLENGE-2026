"""
=============================================================================
LegalIR — Unit Tests
=============================================================================
Kiểm tra các module riêng lẻ mà không cần data thật.
Chạy: python -m pytest tests/ -v
       hoặc: python tests/test_pipeline.py
=============================================================================
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_ir.data_loader import clean_text, normalize_unicode, preprocess
from legal_ir.bm25_retriever import BM25Retriever
from legal_ir.hybrid_ranker import (
    hybrid_score,
    select_top_docs,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_f1_at_k,
    evaluate,
)
from legal_ir.exporter import (
    build_submission_dict,
    validate_predictions,
    export_submission,
)


# ---------------------------------------------------------------------------
# Test: Text preprocessing
# ---------------------------------------------------------------------------
class TestTextPreprocessing(unittest.TestCase):

    def test_normalize_unicode(self):
        """NFC normalization không làm mất nội dung."""
        text = "Điều 1 Luật đất đai"
        result = normalize_unicode(text)
        self.assertEqual(len(result) > 0, True)

    def test_clean_text_removes_control_chars(self):
        text = "Điều\x00 1\x08 Luật"
        result = clean_text(text)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x08", result)

    def test_clean_text_collapses_whitespace(self):
        text = "Điều   1    Luật   đất"
        result = clean_text(text)
        self.assertNotIn("  ", result)

    def test_preprocess_lowercase(self):
        text = "ĐIỀU 1 LUẬT ĐẤT ĐAI"
        result = preprocess(text, segmenter="none", lowercase=True)
        self.assertEqual(result, result.lower())

    def test_preprocess_empty(self):
        self.assertEqual(preprocess("", segmenter="none"), "")


# ---------------------------------------------------------------------------
# Test: BM25 Retriever
# ---------------------------------------------------------------------------
class TestBM25Retriever(unittest.TestCase):

    def setUp(self):
        """Tạo corpus nhỏ để test."""
        self.texts = [
            "điều kiện đăng ký kinh doanh hộ cá thể",
            "thủ tục cấp giấy phép xây dựng nhà ở",
            "quy định về bảo hiểm xã hội bắt buộc",
            "luật đất đai sở hữu toàn dân quyền sử dụng",
            "thuế thu nhập cá nhân biểu thuế suất",
        ]
        self.doc_ids = [101, 202, 303, 404, 505]
        self.retriever = BM25Retriever(variant="BM25Okapi")
        self.retriever.build_index(self.texts, self.doc_ids)

    def test_retrieve_returns_correct_count(self):
        results = self.retriever.retrieve("đất đai quyền sử dụng", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_retrieve_returns_tuples(self):
        results = self.retriever.retrieve("bảo hiểm xã hội", top_k=5)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_retrieve_scores_in_range(self):
        """Normalised scores phải nằm trong [0, 1]."""
        results = self.retriever.retrieve("kinh doanh hộ cá thể", top_k=5)
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0 + 1e-6)

    def test_retrieve_relevant_doc_on_top(self):
        """Doc về 'bảo hiểm xã hội' phải nằm đầu khi query 'bảo hiểm'."""
        results = self.retriever.retrieve("bảo hiểm xã hội bắt buộc", top_k=3)
        top_id = results[0][0]
        self.assertEqual(top_id, 303)

    def test_bm25_cache(self):
        """Test save và load cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_bm25.pkl")
            r = BM25Retriever(variant="BM25Okapi", cache_path=cache_path)
            r.build_index(self.texts, self.doc_ids)
            self.assertTrue(os.path.isfile(cache_path))

            # Load từ cache
            r2 = BM25Retriever(variant="BM25Okapi", cache_path=cache_path)
            r2.build_index(self.texts, self.doc_ids)
            results2 = r2.retrieve("đất đai", top_k=2)
            self.assertGreater(len(results2), 0)


# ---------------------------------------------------------------------------
# Test: Hybrid Score Fusion
# ---------------------------------------------------------------------------
class TestHybridScore(unittest.TestCase):

    def test_hybrid_score_fusion(self):
        bm25_results  = [(1, 1.0), (2, 0.8), (3, 0.5)]
        dense_results = [(1, 0.9), (2, 0.7), (4, 0.6)]
        result = hybrid_score(bm25_results, dense_results, alpha=0.4, beta=0.6)

        # Kết quả phải chứa tất cả docs
        result_ids = [did for did, _ in result]
        for did in [1, 2, 3, 4]:
            self.assertIn(did, result_ids)

    def test_hybrid_score_sorted_descending(self):
        bm25_results  = [(1, 1.0), (2, 0.5)]
        dense_results = [(1, 0.9), (2, 0.8)]
        result = hybrid_score(bm25_results, dense_results)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_select_top_docs_max_limit(self):
        ranked = [(i, 1.0 - i * 0.1) for i in range(10)]
        selected = select_top_docs(ranked, max_docs=5, score_threshold=0.0)
        self.assertLessEqual(len(selected), 5)

    def test_select_top_docs_threshold(self):
        ranked = [(1, 0.9), (2, 0.5), (3, 0.1), (4, 0.05)]
        selected = select_top_docs(ranked, max_docs=5, score_threshold=0.2)
        # Chỉ lấy doc có score >= 0.2: docs 1 và 2
        self.assertIn("1", selected)
        self.assertIn("2", selected)
        self.assertNotIn("3", selected)

    def test_select_returns_strings(self):
        ranked = [(101, 0.8), (202, 0.6)]
        selected = select_top_docs(ranked, max_docs=5)
        for did in selected:
            self.assertIsInstance(did, str)


# ---------------------------------------------------------------------------
# Test: Evaluation Metrics
# ---------------------------------------------------------------------------
class TestEvaluationMetrics(unittest.TestCase):

    def test_perfect_recall(self):
        pred = ["1", "2", "3"]
        gt   = ["1", "2", "3"]
        self.assertAlmostEqual(compute_recall_at_k(pred, gt, k=5), 1.0)

    def test_zero_recall(self):
        pred = ["4", "5"]
        gt   = ["1", "2"]
        self.assertAlmostEqual(compute_recall_at_k(pred, gt, k=5), 0.0)

    def test_partial_recall(self):
        pred = ["1", "3", "5"]
        gt   = ["1", "2", "3", "4"]
        # Hits = {1, 3}, recall = 2/4 = 0.5
        self.assertAlmostEqual(compute_recall_at_k(pred, gt, k=5), 0.5)

    def test_precision_perfect(self):
        pred = ["1", "2"]
        gt   = ["1", "2", "3"]
        self.assertAlmostEqual(compute_precision_at_k(pred, gt, k=5), 1.0)

    def test_precision_partial(self):
        pred = ["1", "4", "5"]
        gt   = ["1", "2"]
        # Hits = {1}, precision = 1/3
        self.assertAlmostEqual(
            compute_precision_at_k(pred, gt, k=5), 1 / 3
        )

    def test_f1_harmonic_mean(self):
        pred = ["1", "2", "3"]
        gt   = ["1", "2", "4", "5"]
        r = compute_recall_at_k(pred, gt, k=5)
        p = compute_precision_at_k(pred, gt, k=5)
        expected_f1 = 2 * r * p / (r + p)
        self.assertAlmostEqual(compute_f1_at_k(pred, gt, k=5), expected_f1)

    def test_evaluate_full(self):
        preds = {"Q1": ["1", "2"], "Q2": ["3"]}
        gts   = {"Q1": ["1", "3"], "Q2": ["3", "4"]}
        metrics = evaluate(preds, gts, k=5, verbose=False)
        self.assertIn("Recall@5",    metrics)
        self.assertIn("Precision@5", metrics)
        self.assertIn("F1@5",        metrics)
        self.assertEqual(metrics["violations"], 0)

    def test_evaluate_violation(self):
        """Câu hỏi trả về > 5 IDs phải bị tính score = 0."""
        preds = {"Q1": ["1", "2", "3", "4", "5", "6"]}  # 6 IDs → vi phạm
        gts   = {"Q1": ["1"]}
        metrics = evaluate(preds, gts, k=5, verbose=False)
        self.assertEqual(metrics["violations"], 1)
        self.assertAlmostEqual(metrics["Recall@5"], 0.0)


# ---------------------------------------------------------------------------
# Test: Exporter
# ---------------------------------------------------------------------------
class TestExporter(unittest.TestCase):

    def test_build_submission_dict_format(self):
        predictions = {"Q1": ["101", "202"], "Q2": ["303"]}
        result = build_submission_dict(["Q1", "Q2"], predictions)
        self.assertIn("Q1", result)
        self.assertIn("Q2", result)
        self.assertEqual(result["Q1"]["answer"], ["101", "202"])

    def test_validate_predictions_pass(self):
        preds = {"Q1": ["1", "2", "3"], "Q2": ["4"]}
        # Không raise exception
        validate_predictions(preds, max_docs=5)

    def test_validate_predictions_fail(self):
        preds = {"Q1": ["1", "2", "3", "4", "5", "6"]}  # > 5
        with self.assertRaises(ValueError):
            validate_predictions(preds, max_docs=5)

    def test_export_submission_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {"Q1": ["1", "2"], "Q2": ["3"]}
            zip_path = export_submission(
                query_ids   = ["Q1", "Q2"],
                predictions = predictions,
                output_dir  = tmpdir,
                max_docs    = 5,
            )
            self.assertTrue(os.path.isfile(zip_path))
            # Kiểm tra file JSON bên trong ZIP
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("submission.json", zf.namelist())
                data = json.loads(zf.read("submission.json"))
                self.assertIn("Q1", data)
                self.assertEqual(data["Q1"]["answer"], ["1", "2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
