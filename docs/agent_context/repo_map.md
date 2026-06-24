# Project Map: ViLaborRAG

ViLaborRAG (Vietnamese Labor Code Question Answering with Citation-grounded Retrieval-Augmented Generation) is an advanced RAG-based question answering system specialized in Vietnamese Labor Code 2019. It provides precise citations at the Article/Clause level and integrates a 3-layer refusal defense mechanism to mitigate hallucinations.

---

## 1. Project Purpose

The primary objective of ViLaborRAG is to build a citation-grounded RAG system for the Vietnamese Labor Code 2019 (Law No. `45/2019/QH14`), ensuring:
- **Article/Clause level citation accuracy** to anchor LLM responses in real legislative text.
- **Hallucination control** and a **3-Layer Refusal System** to reject out-of-scope, fine-related, or unsupported questions.
- **Automated evaluation (Ablation study)** measuring Recall, MRR, Citation Accuracy, Faithfulness, and Refusal Accuracy across configurations A to G.
- **Premium User Experience** via a FastAPI backend and a clean Slate-themed Web UI.

---

## 2. Main Folders and Modules

```text
VI-Legal-RAG/
├── app/                       # FastAPI Web Application & API router
│   ├── api/                   # REST API routes (v1 endpoints)
│   │   └── v1/
│   │       ├── endpoints/     # Query handler (query.py) & System status (system.py)
│   │       └── router.py      # Router aggregator
│   ├── core/                  # Dependency management & pipeline lifecycles
│   ├── schemas/               # Pydantic models for validation
│   └── static/                # Single Page Application frontend (HTML, CSS, JS)
├── configs/                   # Configuration files
│   └── settings.yml           # Central settings for models, thresholds, paths
├── data/                      # Structured & raw data storage
│   ├── raw/                   # raw PDF (`luat-lao-dong.pdf`) and fallback text
│   ├── processed/             # structured JSONL corpus and FAISS vector index
│   └── benchmark/             # QA pairs for evaluation
├── src/                       # RAG Pipeline core logic
│   ├── ingest/                # Extract and clean text from raw PDFs
│   ├── chunking/              # Parse law structure into hierarchy
│   ├── retrieval/             # BM25, Dense (BGE-M3), Hybrid (RRF) search
│   ├── reranking/             # Cross-Encoder (bge-reranker-base) ranking
│   ├── generation/            # Gemini API generator using structured JSON output
│   ├── verification/          # Citation check, Faithfulness check, Refusal detection
│   └── pipeline/              # Main coordinator (RAGPipeline)
└── tests/                     # Pytest suite
```

---

## 3. Entry Points

- **FastAPI Application:** [app/main.py](file:///c:/Users/USER/Desktop/NLP_project/VI-Legal-RAG/app/main.py)
  - Exposed routes: `GET /`, `POST /api/v1/query`, `GET /api/v1/system/status`, `GET /api/v1/system/healthz`, `GET /api/v1/system/readyz`.
- **Ablation Evaluation CLI:** [run_evaluation.py](file:///c:/Users/USER/Desktop/NLP_project/VI-Legal-RAG/run_evaluation.py)
  - Executable benchmark script evaluating different pipeline configurations.
- **Model Downloader Script:** [src/download_models.py](file:///c:/Users/USER/Desktop/NLP_project/VI-Legal-RAG/src/download_models.py)
  - Caches embedding and cross-encoder models locally to support offline mode.

---

## 4. Existing Tests

All automated tests are located under the [tests/](file:///c:/Users/USER/Desktop/NLP_project/VI-Legal-RAG/tests/) folder, executing via `pytest`:
- **API Tests:** `tests/test_api.py` (FastAPI router, CORS settings, query schemas)
- **Pipeline & Modules Tests:** `tests/test_pipeline.py`, `tests/test_retrieval.py`, `tests/test_verification.py`, `tests/test_refusal.py`, `tests/test_chunking.py`, `tests/test_ingest.py`, `tests/test_query_expander.py`, etc.
- **Benchmark Validator:** `tests/test_benchmark_validator.py`

---

## 5. Data & Config Files

- **`configs/settings.yml`:** Central repository settings containing paths, model selections, retrieval score thresholds (for Dense, BM25, RRF Hybrid, and Cross-Encoder), and LLM parameters.
- **`data/raw/luat-lao-dong.pdf`:** Source document for the Vietnamese Labor Code 2019.
- **`data/processed/corpus_structured.jsonl`:** Structured database containing metadata (document, law_number, chapter, section, article, clause, text, source_url) for each chunk.

---

## 6. Current Risks & Failures

- **Test Failure in API testing (`test_query_rag_endpoint_bypass_refusal`):**
  - **Issue:** During the bypass test case, the test asserts that `mock_pipeline.refusal_detector` is mutated into `BypassRefusalDetector`. However, `RAGPipeline.answer_question` handles the bypass by instantiating a local `detector` variable instead of mutating the object-level `self.refusal_detector` to prevent thread-safety or race conditions in multi-threaded API requests.
  - **Impact:** Pytest fails with a `500 Internal Server Error` due to the assertion failing inside the mock `side_effect`.
  - **Fix suggestion:** Modify the mock/assertion in `tests/test_api.py` or modify the pipeline to handle bypass differently under non-concurrent context.

---

## 7. Suggested Next Steps

1. **Resolve API test discrepancy:** Align `tests/test_api.py` assertion logic with the safe, thread-safe implementation in `src/pipeline/rag_pipeline.py`.
2. **Review performance benchmarks:** Run `run_evaluation.py` on the sample benchmark to verify performance metrics of all 7 configurations.
