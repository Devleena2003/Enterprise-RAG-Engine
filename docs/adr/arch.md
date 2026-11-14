# Project 1 Architecture: Enterprise Ingestion & Retrieval Engine

## System Overview

`enterprise-rag-engine` is an independent, zero-cost, CPU-friendly document ingestion and hybrid retrieval service built for messy enterprise PDFs (regulations, policies, financial statements, technical docs).

```mermaid
flowchart TB
    PDF["Messy Enterprise PDF"] --> Ext["PyMuPDF Extractor<br/>(Tables -> Markdown, Strip Margins)"]
    Ext --> Chunk["Structure-Aware Chunker<br/>(Sections, Page Provenance, Overlap)"]
    Chunk --> Emb["FastEmbed ONNX<br/>(BAAI/bge-small-en-v1.5, 384-dim)"]
    Emb --> DB[("PostgreSQL 16 + pgvector<br/>HNSW Cosine + STORED tsvector GIN")]

    Query["POST /api/v1/retrieve"] --> QPrep["Date & OR Query Prep"]
    QPrep --> Hybrid["Hybrid Search<br/>(Vector + Full-Text + RRF k=60)"]
    DB --> Hybrid
    Hybrid --> Rerank["FastEmbed Cross-Encoder<br/>(ms-marco-MiniLM-L-6-v2)"]
    Rerank --> Blend["Blended Score + Refusal Gate"]
    Blend --> Cite["Grounded Evidence + Mandatory Citations"]
```
