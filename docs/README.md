# A-R-E-S Mini: An Advanced RAG System

Welcome to A-R-E-S Mini (Architecture for Retrieval and Extraction of Semantics). This project is an open-source, high-performance Retrieval-Augmented Generation (RAG) system designed to solve the common pitfalls of basic RAG implementations.

## The Problem: Out-of-Context Chunks

Many standard RAG systems struggle with a fundamental problem: when you split a long document into small, independent chunks, those chunks lose their original context. This leads to several issues:
-   **Poor Retrieval:** Chunks containing pronouns or implicit references (e.g., "the company," "his discovery") are often missed by semantic search because they lack the specific keywords of the query.
-   **Incomplete Answers:** The full answer to a user's question might be spread across multiple chunks, and retrieving only one provides an incomplete picture.
-   **LLM Hallucinations:** When a Large Language Model (LLM) is given a set of disconnected or out-of-order chunks, it can get confused and generate inaccurate or fabricated information.

A-R-E-S Mini is engineered from the ground up to solve this "out-of-context chunk" problem through a series of advanced, synergistic techniques.

## Key Features

A-R-E-S Mini combines several state-of-the-art techniques into a cohesive, configurable pipeline:

-   **Context-Aware Embeddings:** Uses a **Late Chunking** and **AutoContext Headers** approach to create chunk embeddings that are aware of both their immediate surroundings and the global context of the entire document.
-   **Adaptive Hybrid Search:** Employs a **Query Planner** that uses an LLM to analyze the user's question and dynamically chooses the best retrieval strategy—either a fast, high-level **Hierarchical Search** on document summaries or a deep, **Hybrid Search** combining semantic (dense) and keyword (sparse) vectors.
-   **Multi-Stage Context Refinement:** The retrieval process doesn't stop at the first search. It uses a pipeline of refinements:
    1.  **Reranking:** A powerful cross-encoder model re-scores and re-orders the initial results for higher accuracy.
    2.  **Relevant Segment Extraction (RSE):** An optimization algorithm reconstructs long, coherent segments of text from the top-ranked chunks, even filling in missing pieces from the original document.
    3.  **MMR-based Diversification:** A final pass to ensure the context sent to the LLM is not only relevant but also informationally diverse, avoiding redundancy.
-   **Modular and Configurable:** Every component, from the embedding models to the RSE parameters, is controlled through a single, well-documented configuration file. It supports both high-performance `vLLM` endpoints and local `transformers` models.

## Quick Start

1.  **Setup:** Ensure you have local instances of [vLLM](https://vllm.ai/) and [Qdrant](https://qdrant.tech/) running.
2.  **Configure:** Copy `configs/ares_mini_config.toml` and modify it to point to your models and services.
3.  **Run:**

```python
from ares_mini.config import load_config
from ares_mini.ingestion.pipeline import create_ingestion_pipeline
from ares_mini.generation.chains import create_rag_chain

# 1. Load configuration
config = load_config()

# 2. Ingest a document
# This pipeline performs all the advanced ingestion steps automatically
ingestion_pipeline = create_ingestion_pipeline(config)
ingestion_pipeline.invoke("path/to/your/document.pdf")

# 3. Create the RAG chain and ask a question
rag_chain = create_rag_chain(config)
question = "Ask a complex question that requires deep context from your document."
answer = rag_chain.invoke(question)

print(answer)
```

## Dive Deeper

-   **[1. Architecture Overview](./1_architecture.md)**: Understand the design philosophy and high-level workflow.
-   **[2. The Ingestion Pipeline](./2_ingestion_pipeline.md)**: A deep dive into how documents are processed and contextualized.
-   **[3. The Retrieval Pipeline](./3_retrieval_pipeline.md)**: Learn how the system adaptively retrieves and refines information.
-   **[4. Configuration Guide](./4_configuration.md)**: A complete reference for all parameters in the `.toml` file.