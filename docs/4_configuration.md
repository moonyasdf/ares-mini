# 4. Configuration Guide

The entire behavior of A-R-E-S Mini is controlled via the `ares_mini_config.toml` file.

## `[generation_llm]`

Configures the primary LLM for generation tasks (planning, summarizing, answering).

-   `provider` (str): `"vllm"` or `"local_transformers"`.
-   `model_name` (str): Hugging Face identifier (e.g., `"Qwen/Qwen3-0.6B-Chat"`).
-   `api_base` (str): Required for `vllm`, the server URL.

## `[embedding_model]`

Configures the model for dense (semantic) embeddings.

-   `provider` (str): `"vllm"` or `"local_transformers"`. `local_transformers` is recommended for models like Qwen3 that need specific logic.
-   `model_name` (str): Hugging Face identifier (e.g., `"Qwen/Qwen3-Embedding-0.6B"`).

## `[reranker_model]`

Configures the cross-encoder model for reranking.

-   `provider` (str): `"vllm"` or `"local_transformers"`. `local_transformers` is recommended for Qwen3's specific reranking logic.
-   `model_name` (str): Hugging Face identifier (e.g., `"Qwen/Qwen3-Reranker-0.6B"`).

## `[sparse_embedder]`

-   `model_name` (str): Model for sparse vectors via `fastembed` (e.g., `"Qdrant/bm25"`).

## `[vector_store]` & `[kv_store]`

Configure the databases for vector and text storage.

-   `provider` (str): `"qdrant"` or `"sqlite"`.
-   `host`, `port`, `collection_name`, `path`: Connection details.

## `[retrieval]`

Controls the query-time retrieval pipeline.

-   `enable_sparse_search` (bool): Toggles the lexical part of the hybrid search.
-   `combination_method` (str): Fusion method: `"rrf"` (recommended) or `"weighted_sum"`.
-   `dense_weight` & `sparse_weight` (float): Weights for the `"weighted_sum"` method.
-   `similarity_top_k` (int): Number of initial candidates to fetch.
-   `rerank_top_n` (int): Number of candidates to keep after reranking.

### `[retrieval.rse]`

Controls Relevant Segment Extraction.

-   `use_rse` (bool): Enables or disables this step.
-   `max_segment_length` (int): Max number of chunks in one segment.
-   `overall_max_segments` (int): Max number of final segments to generate.
-   `min_segment_value` (float): Score threshold for a segment to be considered valid.

### `[retrieval.mmr]`

Controls the final diversification step.

-   `use_mmr` (bool): Enables or disables this step.
-   `final_context_chunks` (int): Final number of segments to send to the LLM.
-   `lambda_mult` (float): `1.0` for max relevance, `0.0` for max diversity.

## `[ingestion]`

-   `chunk_size` & `chunk_overlap` (int): Parameters for `ChonkieTextSplitter`.
-   `generate_descriptions` (bool): Enables title/summary generation, which is required for Hierarchical Search.