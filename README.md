# A-R-E-S Mini: An Advanced, Open-Source RAG System

<p align="center">
  <img src="https://path-to-your-logo.svg" alt="A-R-E-S Mini Logo" width="150"/>
</p>

<p align="center">
  <strong>An open-source, high-performance Retrieval-Augmented Generation (RAG) system built for accuracy, speed, and flexibility.</strong>
</p>

<p align="center">
  <a href="#the-problem-out-of-context-chunks">Why A-R-E-S Mini?</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#architecture-overview">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="./docs/README.md">Full Documentation</a>
</p>

---

A-R-E-S Mini (Architecture for Retrieval and Extraction of Semantics) is an open-source RAG system designed to overcome the limitations of basic RAG pipelines. It integrates a suite of advanced techniques into a streamlined, maintainable, and highly configurable framework powered by **LangChain Expression Language (LCEL)**.

This project was born from the need for an open-source version of the powerful, proprietary ARES system, aiming to deliver its full functional capabilities in a smaller, more accessible codebase.

## The Problem: Out-of-Context Chunks

Standard RAG systems often fail in real-world scenarios because they treat documents as a disconnected bag of chunks. This "out-of-context" problem leads to:

-   **Poor Retrieval:** Chunks with pronouns or implicit references (e.g., *"the city"*, *"their findings"*) are often missed by search because they lack specific keywords.
-   **Incomplete Answers:** The full answer to a user's question is often spread across multiple chunks, but the system only retrieves one or two.
-   **LLM Hallucinations:** When a Large Language Model (LLM) is fed a stream of disconnected or out-of-order text fragments, it can get confused and generate inaccurate information.

**A-R-E-S Mini is engineered to solve this problem**, ensuring that the context provided to the LLM is not just relevant, but also coherent and complete.

## Key Features

A-R-E-S Mini combines several state-of-the-art techniques into a single, cohesive pipeline:

-   ✅ **Context-Aware Embeddings:** Utilizes a **Late Chunking** and **AutoContext Headers** approach. This creates embeddings that understand both the global context of the document and the specific meaning of each chunk, dramatically improving retrieval accuracy.
-   ✅ **Adaptive Hybrid Search:** A **Query Planner** uses an LLM to analyze the user's question and dynamically chooses the best retrieval strategy—either a fast, high-level **Hierarchical Search** on document summaries or a deep, **Hybrid Search** combining semantic (dense) and keyword (sparse) vectors.
-   ✅ **Multi-Stage Context Refinement:** Retrieval is not a single step. A-R-E-S Mini refines the context through a sophisticated pipeline:
    1.  **Reranking:** A powerful cross-encoder model re-scores and re-orders initial results for higher accuracy.
    2.  **Relevant Segment Extraction (RSE):** An optimization algorithm reconstructs long, coherent segments of text from the top-ranked chunks, even filling in missing "sandwiched" chunks from the original document.
    3.  **MMR-based Diversification:** A final pass ensures the context sent to the LLM is informationally rich and non-redundant.
-   ✅ **Flexible Model Providers:** Key components like embedding and reranking models are not hard-coded. You can easily switch between a centralized `vLLM` server and direct, local model loading via `local_transformers` through a single line in the configuration file.
-   ✅ **High-Performance Components:** Leverages best-in-class open-source tools like **Qdrant** for native hybrid search, **Chonkie** for high-speed chunking, and **FastEmbed** for efficient sparse vector generation.

## Architecture Overview

A-R-E-S Mini operates in two main phases, orchestrated by LangChain Expression Language (LCEL) for maximum transparency and modularity.

### 1. Ingestion Pipeline

This pipeline converts raw documents into an optimized, multi-layered knowledge base.

![Ingestion Flow](https://i.imgur.com/your_ingestion_diagram.png) <!-- It is highly recommended to create and link to a real diagram -->
`File -> Load & Describe -> Chunk -> Hybrid & Contextual Embed -> Store`

### 2. Retrieval & Generation Pipeline

This adaptive pipeline processes a user's query through a multi-stage refinement process to build the best possible context before generating an answer.

![Retrieval Flow](https://i.imgur.com/your_retrieval_diagram.png) <!-- It is highly recommended to create and link to a real diagram -->
`Query -> Plan & Route -> Retrieve -> Rerank -> RSE -> MMR -> Generate`

> For a complete technical breakdown, see the full **[Architecture Documentation](./docs/1_architecture.md)**.

## Quick Start

### 1. Prerequisites

-   An running instance of [**Qdrant**](https://qdrant.tech/documentation/quick-start/).
-   An OpenAI-compatible API server like [**vLLM**](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) serving the models specified in the config.
-   Python 3.9+

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/ares-mini.git
cd ares-mini
pip install -r requirements.txt
```

### 3. Configuration

Copy the example configuration file and edit it to match your setup (e.g., update model names, API endpoints).

```bash
cp configs/ares_mini_config.toml.example configs/ares_mini_config.toml
# Now, edit configs/ares_mini_config.toml
```

### 4. Running A-R-E-S Mini

Use the following pattern to ingest documents and ask questions:

```python
from ares_mini.config import load_config
from ares_mini.ingestion.pipeline import create_ingestion_pipeline
from ares_mini.generation.chains import create_rag_chain

# 1. Load configuration from your .toml file
config = load_config("configs/ares_mini_config.toml")

# 2. Ingest a directory of documents
# This pipeline performs all advanced ingestion steps automatically
ingestion_pipeline = create_ingestion_pipeline(config)
ingestion_pipeline.invoke("path/to/your/documents_directory/")

# 3. Create the RAG chain and ask a question
rag_chain = create_rag_chain(config)
question = "Ask a complex question that requires deep context from your documents."
answer = rag_chain.invoke(question)

print(answer)
```

## Documentation

For a deep dive into every component, feature, and configuration option, please see the full documentation in the `/docs` directory.

-   **[1. Architecture Overview](./docs/1_architecture.md)**
-   **[2. The Ingestion Pipeline](./docs/2_ingestion_pipeline.md)**
-   **[3. The Retrieval Pipeline](./docs/3_retrieval_pipeline.md)**
-   **[4. Configuration Guide](./docs/4_configuration.md)**

## Contributing

Contributions are welcome! Whether it's bug fixes, feature proposals, or documentation improvements, please feel free to open an issue or submit a pull request.