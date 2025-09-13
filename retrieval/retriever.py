# --- START OF FILE ares-mini/retrieval/retriever.py ---

import asyncio
from typing import List, Any, Optional, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
import qdrant_client
from qdrant_client.http import models as rest
import numpy as np
from ..config import AresMiniConfig
from ..custom_components import VLLMEmbeddings
from ..ingestion.embedder_sparse import FastEmbedSparseEmbedder

def _normalize_scores(hits: List[rest.ScoredPoint]) -> None:
    """
    Normaliza los scores de una lista de resultados de Qdrant in-place (Min-Max).
    Esto es necesario para la fusión por suma ponderada.
    """
    if not hits:
        return
    scores = np.array([hit.score for hit in hits])
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score == min_score:
        # Evitar división por cero si todos los scores son iguales
        normalized_scores = np.ones_like(scores, dtype=float)
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)
    
    # Actualizar los scores en la lista original de hits
    for hit, norm_score in zip(hits, normalized_scores):
        hit.score = norm_score

class CustomQdrantRetriever(BaseRetriever, BaseModel):
    """
    Retriever personalizado que realiza búsqueda híbrida configurable con dos métodos de fusión:
    Reciprocal Rank Fusion (RRF) o Suma Ponderada. También soporta búsqueda jerárquica a través de filtros.
    """
    config: AresMiniConfig
    embeddings: VLLMEmbeddings
    sparse_embedder: FastEmbedSparseEmbedder = Field(default=None, exclude=True)
    client: qdrant_client.QdrantClient = Field(default=None, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.client = qdrant_client.QdrantClient(host=self.config.vector_store.host, port=self.config.vector_store.port)
        self.sparse_embedder = FastEmbedSparseEmbedder(model_name=self.config.sparse_embedder.model_name)
        print(f"CustomQdrantRetriever (Método: {self.config.retrieval.combination_method}) inicializado.")

    async def _generate_query_embeddings(self, query: str) -> (List[float], Optional[Dict]):
        """Genera embeddings densos y dispersos para una consulta en paralelo."""
        dense_task = asyncio.to_thread(self.embeddings.embed_query, query)
        sparse_task = self.sparse_embedder.aembed_query(query)
        dense_vec, sparse_vec_dict = await asyncio.gather(dense_task, sparse_task)
        return dense_vec, sparse_vec_dict

    async def _dense_search(self, vector: List[float], k: int, filters: Optional[rest.Filter] = None) -> List[rest.ScoredPoint]:
        """Realiza una búsqueda vectorial densa en Qdrant."""
        return await asyncio.to_thread(
            self.client.search,
            collection_name=self.config.vector_store.collection_name,
            query_vector=vector,
            query_filter=filters,
            limit=k,
            with_payload=True
        )

    async def _sparse_search(self, vector_dict: Dict, k: int, filters: Optional[rest.Filter] = None) -> List[rest.ScoredPoint]:
        """Realiza una búsqueda vectorial dispersa en Qdrant."""
        sparse_vector = rest.SparseVector(indices=vector_dict["indices"], values=vector_dict["values"])
        # Asumiendo que el nombre del vector disperso en Qdrant es 'sparse_vectors'
        named_sparse_vector = rest.NamedSparseVector(name="sparse_vectors", vector=sparse_vector)
        return await asyncio.to_thread(
            self.client.search,
            collection_name=self.config.vector_store.collection_name,
            query_vector=named_sparse_vector,
            query_filter=filters,
            limit=k,
            with_payload=True
        )

    async def _hybrid_search_rrf(self, dense_vec, sparse_vec_dict, k, filters) -> List[rest.ScoredPoint]:
        """Realiza una búsqueda híbrida usando Reciprocal Rank Fusion (RRF)."""
        print("--- Ejecutando Búsqueda Híbrida (Fusión: RRF) ---")
        dense_hits_task = self._dense_search(dense_vec, k, filters)
        sparse_hits_task = self._sparse_search(sparse_vec_dict, k, filters)
        dense_results, sparse_results = await asyncio.gather(dense_hits_task, sparse_hits_task)
        
        # Fusión RRF Manual
        ranked_lists = [
            {hit.id: rank + 1 for rank, hit in enumerate(dense_results)},
            {hit.id: rank + 1 for rank, hit in enumerate(sparse_results)}
        ]
        
        rrf_scores, all_ids = {}, set()
        for ranked_list in ranked_lists:
            all_ids.update(ranked_list.keys())
        
        # Constante k de RRF, a menudo se establece en 60
        k_rrf = getattr(self.config.retrieval, "rrf_k", 60)
        for doc_id in all_ids:
            score = sum(1.0 / (k_rrf + ranked_list.get(doc_id, k * 2)) for ranked_list in ranked_lists)
            rrf_scores[doc_id] = score
            
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        if not sorted_ids:
            return []

        # Recuperar los documentos completos para los IDs clasificados por RRF
        hits = await asyncio.to_thread(self.client.retrieve, collection_name=self.config.vector_store.collection_name, ids=sorted_ids, with_payload=True)
        # Asignar el score RRF calculado para mantener el orden y la relevancia
        for hit in hits:
            hit.score = rrf_scores.get(hit.id, 0.0)
        return sorted(hits, key=lambda h: h.score, reverse=True)

    async def _hybrid_search_weighted_sum(self, dense_vec, sparse_vec_dict, k, filters) -> List[rest.ScoredPoint]:
        """Realiza una búsqueda híbrida usando una suma ponderada de scores normalizados."""
        print("--- Ejecutando Búsqueda Híbrida (Fusión: Suma Ponderada) ---")
        dense_hits_task = self._dense_search(dense_vec, k, filters)
        sparse_hits_task = self._sparse_search(sparse_vec_dict, k, filters)
        dense_results, sparse_results = await asyncio.gather(dense_hits_task, sparse_hits_task)

        # Normalizar scores de ambas búsquedas para que estén en la misma escala (0 a 1)
        _normalize_scores(dense_results)
        _normalize_scores(sparse_results)
        
        # Combinar scores usando los pesos de la configuración
        combined_scores = {}
        all_hits_map = {hit.id: hit for hit in dense_results + sparse_results}

        dense_weight = self.config.retrieval.dense_weight
        sparse_weight = self.config.retrieval.sparse_weight

        for hit in dense_results:
            combined_scores[hit.id] = hit.score * dense_weight
        for hit in sparse_results:
            combined_scores[hit.id] = combined_scores.get(hit.id, 0.0) + (hit.score * sparse_weight)
        
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
        
        # Construir la lista final de hits ordenados
        final_hits = []
        for doc_id in sorted_ids:
            hit = all_hits_map[doc_id]
            hit.score = combined_scores[doc_id] # Asignar el nuevo score combinado
            final_hits.append(hit)
            
        return final_hits

    def _format_hits_to_documents(self, hits: List[rest.ScoredPoint]) -> List[Document]:
        """Convierte una lista de ScoredPoint de Qdrant a una lista de Documentos de LangChain."""
        docs = []
        for hit in hits:
            if not hit.payload:
                continue
            metadata = hit.payload.get("metadata", {})
            metadata["_score"] = hit.score
            docs.append(Document(page_content=hit.payload.get("page_content", ""), metadata=metadata))
        return docs
    
    async def asearch_descriptions(self, query: str) -> List[Document]:
        """Busca específicamente en los documentos de descripción (solo denso)."""
        k = self.config.retrieval.similarity_top_k // 2 
        query_vector = self.embeddings.embed_query(query)
        description_filter = rest.Filter(must=[rest.FieldCondition(key="metadata.is_description", match=rest.MatchValue(value=True))])
        
        hits = await self._dense_search(query_vector, k, filters=description_filter)
        return self._format_hits_to_documents(hits)

    async def _aget_relevant_documents(
        self, query: str, *, 
        run_manager: CallbackManagerForRetrieverRun, 
        filters: Optional[rest.Filter] = None, 
        **kwargs
    ) -> List[Document]:
        """Método principal para la recuperación de documentos, orquestando la lógica de búsqueda."""
        k = self.config.retrieval.similarity_top_k
        
        # Combinar filtro base (excluir descripciones) con filtros adicionales (p. ej., de búsqueda jerárquica)
        base_filter = rest.Filter(must_not=[rest.FieldCondition(key="metadata.is_description", match=rest.MatchValue(value=True))])
        if filters:
            if filters.must: base_filter.must = (base_filter.must or []) + filters.must
            if filters.should: base_filter.should = (base_filter.should or []) + filters.should
            if filters.must_not: base_filter.must_not = (base_filter.must_not or []) + filters.must_not
        
        dense_query_vector, sparse_query_vector_dict = await self._generate_query_embeddings(query)

        if not self.config.retrieval.enable_sparse_search or not sparse_query_vector_dict:
            hits = await self._dense_search(dense_query_vector, k, filters=base_filter)
        else:
            method = self.config.retrieval.combination_method
            if method == "rrf":
                hits = await self._hybrid_search_rrf(dense_query_vector, sparse_query_vector_dict, k, filters=base_filter)
            elif method == "weighted_sum":
                hits = await self._hybrid_search_weighted_sum(dense_query_vector, sparse_query_vector_dict, k, filters=base_filter)
            else:
                raise ValueError(f"Método de combinación desconocido: {method}")

        return self._format_hits_to_documents(hits)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs) -> List[Document]:
        """Versión síncrona para compatibilidad con LangChain."""
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager, **kwargs))

# --- END OF FILE ares-mini/retrieval/retriever.py ---