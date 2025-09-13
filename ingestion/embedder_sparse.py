# --- START OF FILE ares-mini/ingestion/embedder_sparse.py ---

from typing import List, Dict, Any, Optional
import asyncio
from fastembed import SparseTextEmbedding, SparseEmbedding

class FastEmbedSparseEmbedder:
    """Wrapper para FastEmbed que genera embeddings dispersos para búsqueda léxica."""
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        try:
            self.model = SparseTextEmbedding(model_name=self.model_name)
            print(f"FastEmbedSparseEmbedder inicializado con el modelo: '{self.model_name}'.")
        except Exception as e:
            print(f"ERROR: No se pudo inicializar FastEmbedSparseEmbedder '{self.model_name}': {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Genera embeddings dispersos para un lote de textos de forma asíncrona."""
        if not texts: return []
        try:
            sparse_embeddings: List[SparseEmbedding] = await asyncio.to_thread(
                list, self.model.embed(texts)
            )
            results = [
                {"indices": se.indices.tolist(), "values": se.values.tolist()} if se and se.indices.size > 0 else None
                for se in sparse_embeddings
            ]
            return results
        except Exception as e:
            print(f"Error generando embeddings dispersos con FastEmbed: {e}")
            return [None] * len(texts)

    async def aembed_query(self, text: str) -> Optional[Dict[str, Any]]:
        """Genera un embedding disperso para una única consulta."""
        results = await self.aembed_documents([text])
        return results[0] if results else None