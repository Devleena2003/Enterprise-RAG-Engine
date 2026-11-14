# Enterprise Ingestion & Retrieval Engine (Project 1)

Production-oriented, $0-cost document ingestion and hybrid retrieval microservice with mandatory page-level citations and strict insufficient-evidence refusal.

## Capabilities

- **Messy PDF Parsing:** Strips repeating headers/footers/page numbers, repairs hyphenation, converts grid tables to Markdown, and rejects scanned/corrupted PDFs explicitly.
- **Structure-Aware Chunking:** Tracks `page_number`, `page_end`, `section_title`, and `chunk_type` (`text` | `table`).
- **Idempotent Ingestion:** SHA-256 `file_hash` deduplication prevents duplicate chunk inflation.
- **Hybrid Search + Reranking:** Combines `pgvector` HNSW cosine similarity and PostgreSQL `tsvector` GIN full-text search via Reciprocal Rank Fusion (RRF), plus ONNX cross-encoder reranking and date-format query expansion.
- **Strict Citations:** Every result includes `document_name`, `page_number`, `section_title`, `chunk_id`, and score breakdown. Returns `sufficient_evidence: false` when evidence falls below threshold.

## Quick Start (Local Development)

1. **Configure environment:**
   ```powershell
   Copy-Item .env.example .env
   ```
