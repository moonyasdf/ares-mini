# --- START OF FILE ares-mini/config.py ---

import toml
from pydantic import BaseModel, Field
from typing import Optional, Literal

# --- Modelos de Configuración Detallados ---

class LLMConfig(BaseModel):
    """Configuración para los modelos de lenguaje y embedding (vLLM)."""
    provider: str
    api_base: str
    api_key: str = "EMPTY"
    model_name: str
    embedding_model: str
    rerank_model: Optional[str] = None

class SparseEmbedderConfig(BaseModel):
    """Configuración para el modelo de vectores dispersos."""
    model_name: str

class VectorStoreConfig(BaseModel):
    """Configuración para la base de datos vectorial (Qdrant)."""
    provider: str
    host: Optional[str] = "localhost"
    port: Optional[int] = 6333
    collection_name: str

class KVStoreConfig(BaseModel):
    """Configuración para el almacén Key-Value (SQLite)."""
    provider: str
    path: str

class RSEConfig(BaseModel):
    """Configuración para Relevant Segment Extraction (RSE)."""
    use_rse: bool
    max_segment_length: int
    overall_max_segments: int
    min_segment_value: float

class MMRConfig(BaseModel):
    """Configuración para Maximal Marginal Relevance (MMR)."""
    use_mmr: bool
    final_context_chunks: int
    lambda_mult: float

class GenerationLLMConfig(BaseModel):
    provider: Literal["vllm", "local_transformers"]
    model_name: str
    api_base: Optional[str] = None
    api_key: Optional[str] = "EMPTY"
    enable_thinking: bool = True 

class EmbeddingConfig(BaseModel):
    provider: Literal["vllm", "local_transformers"]
    model_name: str

class RerankerConfig(BaseModel):
    provider: Literal["vllm", "local_transformers"]
    model_name: str
    enable_thinking: bool = False

class VLLMServerConfig(BaseModel):
    enabled: bool = False
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    # --- NUEVO: Para el "Hard Switch" ---
    chat_template: Optional[str] = None

class RetrievalConfig(BaseModel):
    """
    Configuración completa para el pipeline de recuperación.
    Esta es la versión unificada y correcta.
    """
    enable_sparse_search: bool
    combination_method: Literal["rrf", "weighted_sum"] = "rrf"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    similarity_top_k: int
    rerank_top_n: int
    rse: RSEConfig
    mmr: MMRConfig

class IngestionConfig(BaseModel):
    """Configuración para el pipeline de ingesta."""
    chunk_size: int
    chunk_overlap: int
    generate_descriptions: bool

# --- Modelo de Configuración Raíz ---

class AresMiniConfig(BaseModel):
    """Modelo raíz actualizado."""
    vllm_server: VLLMServerConfig
    generation_llm: GenerationLLMConfig
    embedding_model: EmbeddingConfig
    reranker_model: RerankerConfig
    sparse_embedder: SparseEmbedderConfig
    vector_store: VectorStoreConfig
    kv_store: KVStoreConfig
    retrieval: RetrievalConfig
    ingestion: IngestionConfig

# --- Función de Carga ---

def load_config(config_path: str = "configs/ares_mini_config.toml") -> AresMiniConfig:
    """
    Carga y valida la configuración de la aplicación desde un archivo TOML.
    """
    try:
        with open(config_path, "r") as f:
            config_data = toml.load(f)
        # Pydantic validará automáticamente que la estructura del TOML coincida
        # con el modelo AresMiniConfig y sus sub-modelos.
        return AresMiniConfig(**config_data)
    except FileNotFoundError:
        print(f"ERROR: Archivo de configuración no encontrado en: {config_path}")
        raise
    except Exception as e:
        # Esto capturará errores de validación de Pydantic y otros problemas.
        print(f"ERROR: Error al cargar o validar la configuración: {e}")
        raise