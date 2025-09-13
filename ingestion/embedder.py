# --- START OF FILE ares-mini/ingestion/embedder.py ---

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from transformers import AutoTokenizer
from ..custom_components import VLLMWrapper, ChonkieTextSplitter

class LateChunkingEmbedder:
    """
    Implementa la lógica de Late Chunking inspirada en ARES.
    """
    def __init__(self, vllm_wrapper: VLLMWrapper, config: AresMiniConfig):
        self.vllm = vllm_wrapper
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm.embedding_model)
        self.max_length = 8192 # Límite del modelo de embedding

    def _get_token_spans_for_char_spans(self, text: str, chunk_boundaries: List[Dict]) -> List[Optional[Tuple[int, int]]]:
        """Mapea los límites de caracteres de los chunks a límites de tokens."""
        encoding = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        offset_mapping = encoding.offset_mapping
        
        token_spans = []
        for boundary in chunk_boundaries:
            start_char, end_char = boundary['span']
            start_token_idx, end_token_idx = -1, -1
            
            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if start_token_idx == -1 and offset_start <= start_char < offset_end:
                    start_token_idx = i
                if end_token_idx == -1 and offset_start < end_char <= offset_end:
                    end_token_idx = i + 1
            
            if start_token_idx != -1 and end_token_idx != -1:
                token_spans.append((start_token_idx, end_token_idx))
            else:
                token_spans.append(None)
        return token_spans

    async def embed_document_with_late_chunking(self, document_text: str, headers: List[str]) -> Dict[str, np.ndarray]:
        """Orquesta el proceso de Late Chunking para un documento, incluyendo headers."""
        splitter = ChonkieTextSplitter(
            chunk_size=self.config.ingestion.chunk_size,
            chunk_overlap=self.config.ingestion.chunk_overlap
        )
        
        # Obtener chunks con sus límites en caracteres
        raw_chunks = splitter.chunker.chunk(document_text) # Devuelve objetos ChonkieChunk
        chunk_boundaries = [{"text": c.text, "span": (c.start_index, c.end_index)} for c in raw_chunks]

        # Construir el texto completo para el embedding (Header + Texto)
        header_text = "".join(h + "\n\n" for h in headers)
        text_to_embed = header_text + document_text

        # Obtener embeddings de tokens para el texto completo
        token_embeddings = await self.vllm.aget_token_embeddings(text_to_embed)
        if token_embeddings is None:
            return {}

        # Mapear los límites de caracteres de los chunks originales a los límites de tokens
        # en el `text_to_embed` (que ahora tiene un prefijo de header).
        adjusted_boundaries = [{"text": b["text"], "span": (b["span"][0] + len(header_text), b["span"][1] + len(header_text))} for b in chunk_boundaries]
        token_spans = self._get_token_spans_for_char_spans(text_to_embed, adjusted_boundaries)

        # Aplicar mean pooling
        chunk_embeddings = {}
        for i, span in enumerate(token_spans):
            chunk_text = chunk_boundaries[i]["text"]
            if span:
                start_idx, end_idx = span
                if end_idx <= token_embeddings.shape[0]:
                    chunk_token_embs = token_embeddings[start_idx:end_idx]
                    chunk_emb = np.mean(chunk_token_embs, axis=0)
                    # Usamos el texto del chunk como clave para mapear el embedding
                    chunk_embeddings[chunk_text] = chunk_emb
        
        return chunk_embeddings