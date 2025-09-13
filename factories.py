# --- START OF FILE ares-mini/factories.py ---

from .retrieval.transformers import VLLMReranker
from .config import AresMiniConfig
from .custom_components import (
    VLLMWrapper, VLLMEmbeddings,
    QwenLocalEmbedder, QwenLocalReranker
)

def create_llm(config: AresMiniConfig):
    """Crea la instancia del LLM de generación principal."""
    cfg = config.generation_llm
    if cfg.provider == "vllm":
        return VLLMWrapper(llm_config=cfg)
    # Aquí se podrían añadir otros proveedores locales si fuera necesario
    raise ValueError(f"Proveedor de LLM de generación no soportado: {cfg.provider}")

def create_embedder(config: AresMiniConfig):
    """Crea la instancia del modelo de embedding."""
    cfg = config.embedding_model
    if cfg.provider == "vllm":
        # Asume que la config de generación tiene la info de la API
        llm_cfg = config.generation_llm.model_copy(deep=True)
        llm_cfg.embedding_model = cfg.model_name
        return VLLMEmbeddings(llm_config=config.generation_llm)
    elif cfg.provider == "local_transformers":
        return QwenLocalEmbedder(model_name=cfg.model_name)
    raise ValueError(f"Proveedor de embedding no soportado: {cfg.provider}")

def create_reranker(config: AresMiniConfig, llm: VLLMWrapper):
    """Crea la instancia del reranker."""
    cfg = config.reranker_model
    if cfg.provider == "vllm":
        llm.llm_config.rerank_model = cfg.model_name
        return VLLMReranker(llm=llm, top_n=config.retrieval.rerank_top_n)
    elif cfg.provider == "local_transformers":
        return QwenLocalReranker(model_name=cfg.model_name)
    raise ValueError(f"Proveedor de reranker no soportado: {cfg.provider}")