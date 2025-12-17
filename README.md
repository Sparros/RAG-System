## Retrieval Evaluation

This project includes a systematic, retrieval-only evaluation of embedding models using standard information-retrieval metrics.  
All experiments are run **without LLM generation enabled**, ensuring that results reflect **embedding quality, chunking, and vector indexing only**.

---

## Evaluation Objectives

- Measure retrieval quality independently of the LLM  
- Compare embedding models under identical conditions  
- Analyze ranking quality across different retrieval depths (k-sweep)  
- Evaluate both single-document and cross-document queries  
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
- Questions answerable from a single document  
- Test direct semantic matching and ranking accuracy  

### Cross-Document Queries
- Questions requiring information from multiple documents  
- Test semantic breadth and robustness of ranking  

---

## Metrics

- **Precision@k** – Fraction of retrieved documents that are relevant  
- **Recall@k** – Fraction of relevant documents successfully retrieved  
- **MRR (Mean Reciprocal Rank)** – Average rank position of the first relevant document  
- **Top-1 Accuracy** – Whether the first retrieved document is relevant  

---

## Embedding Models Compared

| Model | Description |
|-----|------------|
| `all-MiniLM-L6-v2` | Lightweight, fast sentence-transformer |
| `all-mpnet-base-v2` | Larger model with stronger semantic capacity |
| `BAAI/bge-base-en-v1.5` | Modern retrieval-optimized embedding model |

---

## k-Sweep Evaluation

Retrieval performance was evaluated at multiple values of *k* to understand ranking behavior under different retrieval depths:

- **k = 1** → strict top-rank accuracy  
- **k = 2–3** → realistic RAG retrieval window  
- **k = 5** → broader recall-focused retrieval  

---

## Results Summary

### k = 3 (Recommended Operating Point)

This value balances ranking quality and contextual breadth for RAG systems.

#### Single-Document Queries

| Model | Precision@3 | Recall@3 | MRR | Top-1 Accuracy |
|-----|-------------|----------|-----|---------------|
| MiniLM | 0.591 | **1.00** | 0.742 | 0.545 |
| MPNet | 0.591 | **1.00** | **0.803** | **0.636** |
| BGE | 0.576 | **1.00** | **0.803** | **0.636** |

#### Cross-Document Queries

| Model | Precision@3 | Recall@3 | MRR | Top-1 Accuracy |
|-----|-------------|----------|-----|---------------|
| MiniLM | **0.952** | 0.929 | 0.929 | 0.857 |
| MPNet | **0.952** | 0.929 | **1.000** | **1.000** |
| BGE | **0.952** | 0.929 | 0.929 | 0.857 |

---

## Key Observations

- All models achieved **perfect recall** on single-document queries across all k, validating chunking and indexing correctness.
- **MPNet and BGE consistently ranked relevant documents higher**, reflected in stronger MRR and Top-1 accuracy.
- **MiniLM performed competitively** despite being significantly smaller and faster.
- Cross-document retrieval remained strong across models, indicating effective semantic coverage.
- Increasing *k* improves recall but reduces precision, highlighting the importance of selecting an appropriate retrieval depth.
- Retrieval-only evaluation exposed ranking differences that would be hidden by LLM generation.

---

## Why This Matters

Most RAG systems select embeddings heuristically.  
This project demonstrates a **measurement-driven approach** to embedding selection using:

- Controlled experiments  
- Multiple embedding models  
- k-sweep analysis  
- Standard IR metrics  

The evaluation framework is fully reproducible and extensible, providing a solid foundation for future RAG optimization and research-grade experimentation.

## Robustness Evaluation: Adversarial & Noisy Queries

To assess retrieval robustness beyond idealized queries, the evaluation was extended with **adversarial** and **noisy** query sets.

### Adversarial Queries

Adversarial queries use ambiguous phrasing, underspecified references, or overlapping domain concepts to stress-test semantic retrieval.

Examples:
- Ambiguous entities shared across domains
- Questions omitting key disambiguating terms
- Broad conceptual phrasing instead of factual keywords

### Noisy Queries

Noisy queries simulate real user input variability, including:
- Typos and misspellings
- Paraphrased questions
- Informal or conversational phrasing

---

## Reranking Experiment

A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was evaluated to measure its impact on robustness.

Two configurations were compared:
- **Embedding-only retrieval**
- **Embedding + cross-encoder reranking**

All experiments remain retrieval-only (no LLM generation).

---

## Key Findings

### Embedding-Only Retrieval

- Performed strongly on clean, well-specified queries
- Exhibited **retrieval failures on adversarial queries**, particularly at low k
- Failures were most prominent for ambiguous queries requiring semantic disambiguation

### With Cross-Encoder Reranking

- **Reduced adversarial failures to near-zero at k ≥ 2**
- Improved recall and MRR for ambiguous queries
- Minimal impact on clean single-document and cross-document queries

Reranking did **not** improve performance at k = 1, as rerankers require multiple candidates to reorder.

---

## Interpretation

These results confirm a common RAG systems pattern:

- **Embedding models provide semantic recall**
- **Cross-encoders provide precision under ambiguity**

The reranker is most effective when:
- Queries are underspecified or ambiguous
- Multiple candidate documents are available
- Retrieval depth allows reordering (k ≥ 2)

---

## Implications for RAG System Design

- Embedding-only retrieval is sufficient for clean, factual queries
- Reranking becomes critical in real-world scenarios with noisy or ambiguous user input
- Evaluating only clean queries can significantly overestimate system robustness

This evaluation demonstrates how controlled stress testing reveals failure modes that would otherwise be hidden.



## Limitations & Future Work

While this project demonstrates a robust and evaluation-driven RAG system, several limitations remain and provide clear directions for future improvement.

### Limitations

**1. Small-Scale Corpus**  
The evaluation corpus contains only 10 documents (20 chunks). While this was sufficient to compare embedding behavior across domains, results may not generalize directly to large-scale or highly redundant corpora.

**2. Limited Ground Truth Granularity**  
Relevance labels are defined at the document level rather than the chunk level. This simplifies evaluation but does not capture partial relevance or fine-grained semantic alignment within long documents.

**3. FAISS Flat Index Only**  
The current system uses a brute-force FAISS `IndexFlatL2`. This ensures correctness but does not reflect the performance trade-offs required at larger scales where approximate indexing (HNSW, IVF) becomes necessary.

**4. No Query Reformulation or Reranking**  
Retrieval relies solely on embedding similarity. Advanced techniques such as cross-encoder reranking or query expansion are not currently implemented.

**5. LLM Evaluation Not Quantified**  
While the system supports full RAG generation, hallucination rates and answer faithfulness are not yet formally measured. The current evaluation focuses exclusively on retrieval quality.

---

### Future Work

**1. Larger and Noisier Corpora**  
Expand the corpus to hundreds or thousands of documents to study scalability, robustness, and semantic drift in retrieval.

**2. Chunk-Level Relevance Annotation**  
Introduce fine-grained relevance labels to better evaluate ranking quality and partial matches.

**3. Advanced Indexing Strategies**  
Evaluate approximate nearest neighbor indexes (e.g., FAISS HNSW, IVF) to measure latency–accuracy trade-offs in realistic deployments.

**4. Reranking and Hybrid Retrieval**  
Add cross-encoder rerankers or keyword-based hybrid retrieval to improve precision, especially for multi-hop queries.

**5. LLM Faithfulness Evaluation**  
Quantify hallucination rates using citation accuracy, answer grounding checks, or human-in-the-loop evaluation.

**6. End-to-End RAG Metrics**  
Extend evaluation to include answer-level metrics such as faithfulness, completeness, and source attribution quality.

---

This project is intentionally designed as a **foundation for iterative RAG research**, enabling controlled experimentation and future extension without architectural changes.
