# 2. The Ingestion Pipeline

The ingestion pipeline is designed to be a "fire-and-forget" process that converts entire directories of documents into a highly optimized format for retrieval.

## Step 1: Document Loading & Page Awareness

-   **Loader:** `AresDocumentLoader` (powered by PyMuPDF).
-   **Process:** When a PDF is loaded, it is immediately split into a list of `Document` objects, one for each page.
-   **Key Feature:** This preserves the `page_number` metadata from the very beginning, ensuring every subsequent chunk can be traced back to its exact page.

## Step 2: High-Level Description (for AutoContext & Hierarchical Search)

-   **Component:** `DescriptionGenerator`.
-   **Process:** For each document, its initial text is sent in a batch to a generation LLM, which returns a structured **title** and **summary**.
-   **Storage:** This description is then embedded and stored as a special, separate entry in Qdrant with `is_description: True` metadata. This creates the high-level "summary layer" for hierarchical search.

## Step 3: Chunking

-   **Component:** `ChonkieTextSplitter`.
-   **Process:** The content of *each page* is individually passed to the Chonkie chunker. This granular approach ensures that the `page_number` metadata is correctly associated with every chunk generated from that page.

## Step 4: Hybrid & Contextual Embedding

This is the core of ARES-Mini's ingestion intelligence, combining two crucial techniques.

#### AutoContext Headers

-   **Purpose:** To provide explicit, high-level context to the embedding model.
-   **Process:** Before embedding, a text "header" is constructed using the title and summary generated in Step 2. This header is prepended to the full text of the document.
-   **Example Header:** `Título del Documento: Informe Anual Nike 2023\n\nResumen del Documento: Resultados financieros y operativos...`

#### Late Chunking (for Dense Vectors)

1.  **Combined Text:** The system creates a large in-memory text block: `[AutoContext Header] + [Full Document Text]`.
2.  **Token-level Embeddings:** This entire block is sent to the embedding model to get the `last_hidden_state`, which contains a vector for every single token. Because the entire document and header were processed together, each token's vector is implicitly aware of the full context.
3.  **Contextual Pooling:** The system uses pre-calculated character-to-token mappings to identify which token vectors correspond to each chunk. It then performs a `mean pooling` operation on these specific token vectors to create the final, highly contextual dense embedding for that chunk.

#### Sparse Vector Generation

-   **Component:** `FastEmbedSparseEmbedder`.
-   **Process:** In parallel, the raw text of each chunk (without the header) is processed to generate a sparse vector, capturing its most important keywords.

## Step 5: Storage

-   **Vector Store (Qdrant):** Each chunk is stored as a single point containing its dense vector, sparse vector, and rich metadata (`page_number`, `chunk_id`, etc.).
-   **KV Store (SQLite):** The raw text of each chunk and the document's structural map (the ordered list of all its chunk IDs) are stored. This is critical for the RSE process to be able to reconstruct segments at query time.