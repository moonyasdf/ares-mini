# 3. The Retrieval & Generation Pipeline

The retrieval pipeline is an adaptive LCEL chain that intelligently processes a user's query to deliver the most relevant and complete context to the final LLM.

## Step 1: Query Planning & Routing

-   **Component:** `QueryPlanner` and `RunnableBranch`.
-   **Process:** The user's query is first sent to an LLM, which determines the optimal retrieval strategy. The LCEL chain then uses `RunnableBranch` to route the query down the correct path.
-   **Strategies:**
    -   **Hierarchical Search (`description`):** For broad questions. It first searches the small index of document summaries to find the most relevant documents, and then performs a detailed hybrid search *only* within those documents. This is extremely efficient for large knowledge bases.
    -   **Direct Hybrid Search (`hybrid`):** For specific questions. It directly executes a hybrid search across all chunks in the database.

## Step 2: Hybrid Retrieval

-   **Component:** `CustomQdrantRetriever`.
-   **Process:** This retriever is responsible for all communication with Qdrant.
    -   It generates both a **dense vector** (for semantic meaning) and a **sparse vector** (for keywords) from the user's query.
    -   It sends both to Qdrant in a single request.
    -   It then fuses the results from both searches using a configurable method (`rrf` or `weighted_sum`) to produce a single, robustly ranked list of candidate chunks.

## Step 3: Reranking

-   **Component:** `VLLMReranker` or `QwenLocalReranker`.
-   **Process:** The candidate chunks from the retriever are passed to a more powerful cross-encoder model. This model meticulously re-evaluates the relevance of each chunk against the query and outputs a new, more accurate ranking. This step is crucial for filtering out "false positives" from the initial retrieval.

## Step 4: Relevant Segment Extraction (RSE)

-   **Component:** `RSETransformer`.
-   **Process:** This is the core context reconstruction engine.
    1.  **Grouping:** It groups the reranked chunks by their original document.
    2.  **Filling Gaps:** It queries the **KV Store** to fetch any "sandwiched" chunks—chunks that are located between two highly relevant chunks but were not retrieved initially.
    3.  **Optimization:** It runs an algorithm to find the most valuable contiguous sequence of chunks (a "segment") that maximizes the total relevance score.
    4.  **Merging:** It combines the text of the chunks in this optimal segment into a single, long, and coherent piece of context.

## Step 5: MMR Diversification (Optional)

-   **Component:** `MMRTransformer`.
-   **Process:** If enabled, this final step takes the list of generated segments and selects a subset that is both highly relevant to the query and informationally diverse, preventing the LLM from receiving repetitive information.

## Step 6: Final Answer Generation

-   **Process:** The final, refined context segments are formatted into a prompt template along with the original query. This complete prompt is then sent to the main Generation LLM, which synthesizes a final, source-grounded answer.