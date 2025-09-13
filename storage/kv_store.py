# --- START OF FILE ares-mini/storage/kv_store.py ---

from typing import Dict, Any
# Asumimos que tendremos implementaciones que siguen una interfaz común
# from langchain_community.storage import SQLiteStore, RedisStore, etc.

def get_kv_store(config: Dict[str, Any], db_name: str = "main_kv.db"):
    """Factory function to get an instance of a Key-Value store."""
    # Esta es una implementación simplificada. Una versión robusta usaría
    # importación dinámica o un mapa de clases.
    if config["provider"] == "sqlite":
        # LangChain no tiene un KVStore SQLite directo y simple, así que
        # simulamos la creación de uno. En un proyecto real, se usaría
        # una clase que implemente la interfaz `BaseStore`.
        from langchain_community.storage import SQLiteStore
        # El path debería ser relativo al working_dir, manejado por el llamador.
        return SQLiteStore(db_path=config["path"])
    else:
        raise ValueError(f"Unsupported KV store provider: {config['provider']}")