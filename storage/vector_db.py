# --- START OF FILE ares-mini/storage/vector_db.py ---

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant
import qdrant_client
from qdrant_client.http import models as rest

from ..config import VectorStoreConfig

class QdrantHybridStore(Qdrant, VectorStore):
    """
    Clase extendida para Qdrant que asegura la existencia de una colección híbrida
    y proporciona métodos para la ingesta de vectores densos y dispersos.
    """
    SPARSE_VECTOR_NAME = "sparse_vectors"

    def add_vectors(self, documents: List[Document], vectors: List[Dict[str, Any]]):
        """Añade puntos con vectores densos y dispersos a Qdrant."""
        points = []
        for i, doc in enumerate(documents):
            point_id = doc.metadata.get("chunk_id")
            if not point_id:
                raise ValueError(f"Documento en índice {i} no tiene 'chunk_id' en metadatos.")

            vector_struct = {
                "dense": vectors[i]["dense"].tolist(),
                self.SPARSE_VECTOR_NAME: rest.SparseVector(
                    indices=vectors[i]["sparse"]["indices"],
                    values=vectors[i]["sparse"]["values"]
                )
            }
            
            points.append(rest.PointStruct(
                id=point_id,
                vector=vector_struct,
                payload={"page_content": doc.page_content, "metadata": doc.metadata}
            ))
        
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            print(f"Upserted {len(points)} puntos híbridos en '{self.collection_name}'.")

def get_vector_store(config: VectorStoreConfig, embeddings: Embeddings) -> QdrantHybridStore:
    """Fábrica que inicializa Qdrant y asegura una colección híbrida."""
    client = qdrant_client.QdrantClient(host=config.host, port=config.port)
    
    # Asegurar que la colección exista y sea compatible
    try:
        collection_info = client.get_collection(collection_name=config.collection_name)
        if QdrantHybridStore.SPARSE_VECTOR_NAME not in collection_info.sparse_vectors_config.map:
             raise ValueError(f"La colección '{config.collection_name}' existe pero no tiene configuración para vectores dispersos '{QdrantHybridStore.SPARSE_VECTOR_NAME}'.")
    except Exception as e:
        if "not found" in str(e).lower():
            print(f"Creando nueva colección híbrida: '{config.collection_name}'")
            client.recreate_collection(
                collection_name=config.collection_name,
                vectors_config=rest.VectorParams(
                    size=1024, # Dimensión del modelo BGE-large
                    distance=rest.Distance.COSINE
                ),
                sparse_vectors_config={
                    QdrantHybridStore.SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                        index=rest.SparseIndexParams(on_disk=False)
                    )
                }
            )
        else:
            raise e

    return QdrantHybridStore(
        client=client,
        collection_name=config.collection_name,
        embeddings=embeddings,
    )