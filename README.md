## Retrieval Evaluation

This project includes a systematic evaluation of retrieval quality across multiple embedding models using standard information-retrieval metrics.  
All experiments are run **without LLM generation enabled**, ensuring results reflect embedding and indexing performance only.

---

## Evaluation Objectives

- Measure retrieval quality independently of the LLM  
- Compare embedding models under identical conditions  
- Analyze ranking quality for both simple and multi-document queries  
- Provide reproducible, metric-driven evidence for embedding choice  

---

## Evaluation Corpus

**Total documents:** 10  
**Total chunks indexed:** 20  

The corpus spans multiple domains to prevent embeddings from benefiting purely from topical similarity.

### Domains Covered

- Biology / Medicine  
- Machine Learning  
- Programming (FastAPI)  
- DevOps (Docker)  
- History  
- Finance  
- Law  
- Networking / Protocols  

### Document Size Distribution (Approx.)

| File | Domain | Length |
|-----|--------|--------|
| `bio_aspirin.txt` | Medicine | ~2,300 words |
| `bio_neural_network.txt` | Biology / ML | ~1,600 words |
| `prog_fastapi.txt` | Programming | ~1,200 words |
| `prog_docker.txt` | DevOps | ~1,400 words |
| `history_glorious_revolution.txt` | History | ~1,300 words |
| `history_newton.txt` | History | ~2,500 words |
| `finance_compound_interest.txt` | Finance | ~900 words |
| `finance_inflation.txt` | Finance | ~1,400 words |
| `law_habeas_corpus.txt` | Law | ~1,100 words |
| `tech_http.txt` | Networking | ~2,000 words |

All documents were:

- Sourced from Wikipedia  
- Cleaned (references, tables, navigation removed)  
- Chunked using a fixed-size overlapping window  
- Embedded and indexed using FAISS  

---

## Query Sets

Two query types are evaluated:

### Single-Document Queries

- Factual questions answerable from a single document  
- Tests direct semantic matching  

### Cross-Document Queries

- Questions requiring information from multiple documents  
- Tests semantic breadth and ranking robustness  

---

## Metrics

- **Precision@k** – Fraction of retrieved documents that are relevant  
- **Recall@k** – Fraction of relevant documents successfully retrieved  
- **MRR (Mean Reciprocal Rank)** – Position of the first relevant result  
- **Top-1 Accuracy** – Whether the first retrieved result is relevant  

---

## Embedding Models Compared

| Model | Description |
|-----|------------|
| `all-MiniLM-L6-v2` | Lightweight, fast sentence-transformer |
| `all-mpnet-base-v2` | Larger model with stronger semantic representation |

---

## Results

### Single-Document Queries

| Model | Precision@k | Recall@k | MRR | Top-1 Accuracy |
|-----|-------------|----------|-----|---------------|
| MiniLM | 0.318 | **1.00** | 0.523 | 0.182 |
| MPNet | 0.314 | **1.00** | **0.553** | **0.273** |

---

### Cross-Document Queries

| Model | Precision@k | Recall@k | MRR | Top-1 Accuracy |
|-----|-------------|----------|-----|---------------|
| MiniLM | **0.595** | 0.929 | **0.762** | 0.571 |
| MPNet | 0.557 | 0.929 | **0.762** | 0.571 |

---

## 🔎 Key Observations

- Both models achieved **perfect recall** on single-document queries, validating chunking and indexing.
- MiniLM slightly outperformed MPNet in **precision** on cross-document queries.
- MPNet showed higher **MRR and Top-1 Accuracy** on single-document queries, indicating stronger top-rank ordering.
- Larger embedding models did **not consistently outperform** smaller ones on this compact, multi-domain corpus.
- Separating retrieval evaluation from generation revealed ranking behavior that would otherwise be masked by LLM outputs.

---

## Why This Matters

Most RAG systems select embeddings heuristically.  
This project demonstrates a **measurement-driven approach** to embedding selection using reproducible evaluation, standard IR metrics, and controlled experimental design.
