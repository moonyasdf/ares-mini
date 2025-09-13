# --- START OF FILE ares-mini/retrieval/transformers.py ---

import asyncio
import numpy as np
from typing import Any, Sequence, List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents.transformers import BaseDocumentTransformer
from ..custom_components import VLLMEmbeddings

from ..custom_components import VLLMWrapper
from ..storage.kv_store import get_kv_store # Asumimos una factory para el KV store

class VLLMReranker(BaseDocumentTransformer, BaseModel):
    """
    Reranks documents usando un modelo cross-encoder alojado en vLLM.
    """
    llm: VLLMWrapper
    top_n: int = Field(default=25, description="Número de documentos a devolver después del reranking.")

    class Config:
        arbitrary_types_allowed = True

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return asyncio.run(self.atransform_documents(documents, **kwargs))

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        query = kwargs.get("query")
        if not query:
            raise ValueError("Se debe proporcionar una 'query' para el reranking.")
        if not documents:
            return []

        print(f"--- VLLMReranker: Reordenando {len(documents)} docs para la consulta: '{query[:50]}...' ---")
        doc_texts = [doc.page_content for doc in documents]
        
        scores = await self.llm.arerank(query=query, documents=doc_texts)
        
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = score

        reranked_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        
        print(f"--- VLLMReranker: Top 3 scores: {[f'{d.metadata.get('rerank_score', 0.0):.4f}' for d in reranked_docs[:3]]} ---")
        
        return reranked_docs[:self.top_n]


class RSETransformer(BaseDocumentTransformer, BaseModel):
    """
    Implementación completa de Relevant Segment Extraction (RSE) como un transformador de LangChain.
    """
    chunk_store_config: Dict[str, Any]
    chunk_store: Any = None
    max_segment_length: int = Field(default=10, description="Max chunks en un solo segmento.")
    overall_max_segments: int = Field(default=3, description="Total de segmentos a devolver.")
    min_segment_value: float = Field(default=0.3, description="Valor mínimo para que un segmento sea válido.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.chunk_store = get_kv_store(self.chunk_store_config)
        print("RSETransformer inicializado con KVStore.")

    def _get_chunk_value(self, rank: int, score: float) -> float:
        decay_rate, penalty = 30.0, 0.18
        return np.exp(-rank / decay_rate) * max(0, score) - penalty

    async def _get_full_window_sequence(self, window_id: str, reranked_docs: List[Document]) -> List[Dict[str, Any]]:
        window_meta_key = f"win_meta_{window_id}"
        # La implementación de SQLiteStore de LangChain no es async, usamos to_thread
        window_meta_list = await asyncio.to_thread(self.chunk_store.mget, [window_meta_key])
        window_meta = window_meta_list[0] if window_meta_list else None
        
        if not window_meta or "chunk_ids" not in window_meta:
            return [{"doc": doc, "score": doc.metadata.get("rerank_score", 0.0), "rank": i} for i, doc in enumerate(reranked_docs)]

        all_chunk_ids = window_meta["chunk_ids"]
        full_sequence: List[Optional[Dict]] = [None] * len(all_chunk_ids)
        reranked_map = {doc.metadata["chunk_id"]: doc for doc in reranked_docs}
        
        ids_to_fetch = [chunk_id for chunk_id in all_chunk_ids if chunk_id not in reranked_map]
        fetched_map = {}
        if ids_to_fetch:
            fetched_data = await asyncio.to_thread(self.chunk_store.mget, ids_to_fetch)
            fetched_map = dict(zip(ids_to_fetch, fetched_data))

        for i, chunk_id in enumerate(all_chunk_ids):
            if chunk_id in reranked_map:
                doc = reranked_map[chunk_id]
                full_sequence[i] = {"doc": doc, "score": doc.metadata.get("rerank_score", 0.0), "rank": doc.metadata.get("rerank_rank", 999)}
            elif chunk_id in fetched_map and fetched_map[chunk_id]:
                data = fetched_map[chunk_id]
                doc = Document(page_content=data["text"], metadata=data.get("metadata", {}))
                full_sequence[i] = {"doc": doc, "score": 0.0, "rank": 999}
        
        return [item for item in full_sequence if item is not None]

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        if not documents: return []
        
        print(f"--- RSETransformer: Procesando {len(documents)} documentos rerankeados ---")
        
        for i, doc in enumerate(documents):
            doc.metadata["rerank_rank"] = i

        docs_by_window: Dict[str, List[Document]] = {}
        for doc in documents:
            if "window_id" in doc.metadata and "chunk_id" in doc.metadata:
                docs_by_window.setdefault(doc.metadata.get("window_id"), []).append(doc)

        all_segments = []
        for window_id, window_docs in docs_by_window.items():
            full_sequence = await self._get_full_window_sequence(window_id, window_docs)
            if not full_sequence: continue

            chunk_values = [self._get_chunk_value(item["rank"], item["score"]) for item in full_sequence]
            
            # Algoritmo de optimización para encontrar los mejores segmentos
            window_segments = []
            excluded_indices = set()
            for _ in range(self.overall_max_segments):
                best_segment_value = -np.inf
                best_segment_indices = None
                for start in range(len(full_sequence)):
                    if start in excluded_indices: continue
                    for end in range(start + 1, min(start + self.max_segment_length + 1, len(full_sequence) + 1)):
                        if any(idx in excluded_indices for idx in range(start, end)): continue
                        
                        segment_value = sum(chunk_values[start:end])
                        if segment_value > best_segment_value:
                            best_segment_value = segment_value
                            best_segment_indices = (start, end)
                
                if best_segment_indices and best_segment_value > self.min_segment_value:
                    start_idx, end_idx = best_segment_indices
                    window_segments.append({
                        "score": best_segment_value,
                        "docs": [item["doc"] for item in full_sequence[start_idx:end_idx]]
                    })
                    excluded_indices.update(range(start_idx, end_idx))
                else:
                    break
            all_segments.extend(window_segments)

        sorted_segments = sorted(all_segments, key=lambda x: x["score"], reverse=True)
        
        final_docs = [
            Document(
                page_content="\n\n".join(doc.page_content for doc in seg["docs"]),
                metadata={
                    "source": "RSE segment", "segment_score": seg["score"],
                    "original_window_id": seg["docs"][0].metadata.get("window_id"),
                    "num_chunks": len(seg["docs"])
                }
            ) for seg in sorted_segments[:self.overall_max_segments]
        ]
        
        print(f"--- RSETransformer: Extrajo {len(final_docs)} segmentos de alto valor. ---")
        return final_docs

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return asyncio.run(self.atransform_documents(documents, **kwargs))

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores numpy."""
    if v1 is None or v2 is None: return 0.0
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0: return 0.0
    return np.dot(v1, v2) / (v1_norm * v2_norm)

class MMRTransformer(BaseDocumentTransformer, BaseModel):
    """
    Aplica Maximal Marginal Relevance (MMR) a una lista de documentos
    (o segmentos de RSE) para diversificar los resultados finales.
    """
    embeddings: VLLMEmbeddings
    k: int = Field(default=5, description="Número final de documentos a seleccionar.")
    lambda_mult: float = Field(default=0.6, description="Factor de diversidad vs relevancia.")

    class Config:
        arbitrary_types_allowed = True
    
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        if not documents or self.k <= 0:
            return []
        
        # Embeber todos los documentos/segmentos
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        selected_indices = []
        candidate_indices = list(range(len(documents)))
        
        # El primer documento es siempre el más relevante (ya vienen ordenados por reranker/RSE)
        selected_indices.append(candidate_indices.pop(0))
        
        while len(selected_indices) < self.k and candidate_indices:
            best_score = -float('inf')
            best_idx_to_add = -1
            
            query_embedding = self.embeddings.embed_query(kwargs.get("query", "")) # Necesitamos la query original
            
            for idx in candidate_indices:
                relevance_score = _cosine_similarity(np.array(query_embedding), np.array(doc_embeddings[idx]))
                
                max_similarity_to_selected = -float('inf')
                for sel_idx in selected_indices:
                    sim = _cosine_similarity(np.array(doc_embeddings[idx]), np.array(doc_embeddings[sel_idx]))
                    if sim > max_similarity_to_selected:
                        max_similarity_to_selected = sim

                mmr_score = self.lambda_mult * relevance_score - (1 - self.lambda_mult) * max_similarity_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx_to_add = idx
            
            if best_idx_to_add != -1:
                selected_indices.append(best_idx_to_add)
                candidate_indices.remove(best_idx_to_add)
            else:
                break
                
        return [documents[i] for i in selected_indices]