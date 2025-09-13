# ares-mini/main_test_runner.py

import asyncio
import os
import sys
import time
import shutil
from typing import Optional

# --- Asegurar que el proyecto esté en el path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ares_mini.config import load_config, AresMiniConfig
from ares_mini.generation.chains import create_rag_chain
from ares_mini.ingestion.pipeline import create_ingestion_pipeline
from ares_mini.vllm_manager import VLLMServerManager

# --------------------------------------------------------------------------
# --- CONFIGURACIÓN DE LA PRUEBA ---
# --------------------------------------------------------------------------
CONFIG_FILENAME = "ares_mini_full_control_config.toml" # <- El nuevo config
ARES_CONFIG_PATH: str = os.path.join(script_dir, "configs", CONFIG_FILENAME)

INITIAL_DOCUMENT_PATH = "/home/moony/Documentos/La filosofía de la redención (Philipp Mainländer) (Z-Library).pdf" # <- ¡MODIFICA ESTA RUTA!

CLEAN_DATA_BEFORE_RUN = True

# --- Control de Flujo de la Prueba ---
# Decide qué partes del script ejecutar
DO_INGESTION = True
DO_CHAT = True
# --------------------------------------------------------------------------


async def chat_loop(rag_chain):
    """Bucle principal de chat interactivo con el usuario."""
    print("\n" + "="*20 + " Chat Interactivo con ARES-mini " + "="*20)
    print("La ingesta ha finalizado. ¡Puedes empezar a preguntar!")
    print("Escribe tu consulta o 'exit' para salir.")
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nTú: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo del chat...")
            break

        if user_input.lower().strip() == 'exit':
            break
        if not user_input.strip():
            continue

        try:
            print("\nARES-mini está pensando...")
            start_time = time.perf_counter()
            answer = await rag_chain.ainvoke(user_input)
            end_time = time.perf_counter()
            print(f"\n🤖 ARES-mini (Respuesta en {end_time - start_time:.2f}s):")
            print(answer)
        except Exception as e:
            print(f"\nHa ocurrido un error al procesar tu solicitud: {e}")

async def main():
    """Función principal que inicializa, ingesta y comienza el chat."""
    print("\n" + "="*25 + " Iniciando ARES-mini Test Runner (Control Total) " + "="*25)
    
    config: AresMiniConfig = load_config(ARES_CONFIG_PATH)

    if CLEAN_DATA_BEFORE_RUN:
        kv_store_path = config.kv_store.path
        # Asegurarse de que el directorio padre exista para el path relativo
        kv_dir = os.path.dirname(kv_store_path)
        if kv_dir and not os.path.exists(kv_dir):
            os.makedirs(kv_dir)
            
        if os.path.exists(kv_store_path):
            print(f"ADVERTENCIA: Limpiando KV Store antiguo en '{kv_store_path}'...")
            if os.path.isdir(kv_store_path): shutil.rmtree(kv_store_path)
            else: os.remove(kv_store_path)
        # Limpieza de Qdrant, asume que está en un subdirectorio del kv_store
        qdrant_path = os.path.join(kv_dir, "qdrant_storage")
        if os.path.exists(qdrant_path):
            print(f"ADVERTENCIA: Limpiando Vector Store antiguo en '{qdrant_path}'...")
            shutil.rmtree(qdrant_path)

    vllm_manager = VLLMServerManager(config)
    rag_chain = None

    try:
        # --- GESTIÓN DEL SERVIDOR VLLM ---
        if config.vllm_server.get("enabled", False) and (DO_INGESTION or DO_CHAT):
            await vllm_manager.start_server()

        # --- FASE DE INGESTA (SI ESTÁ HABILITADA) ---
        if DO_INGESTION:
            print("\n--- Fase de Ingesta ---")
            if not os.path.exists(INITIAL_DOCUMENT_PATH):
                print(f"ERROR: La ruta '{INITIAL_DOCUMENT_PATH}' no existe.")
                return

            # INICIALIZACIÓN INTELIGENTE: Solo creamos el pipeline si vamos a ingestar.
            # Esto carga el modelo de embedding localmente.
            ingestion_pipeline = create_ingestion_pipeline(config)
            
            ingest_start_time = time.perf_counter()
            ingestion_pipeline.invoke(INITIAL_DOCUMENT_PATH)
            duration = time.perf_counter() - ingest_start_time
            print(f"\n✅ INGESTA COMPLETADA en {duration:.2f} segundos.")

        # --- FASE DE CHAT (SI ESTÁ HABILITADA) ---
        if DO_CHAT:
            print("\n--- Fase de Generación (Chat) ---")
            
            # INICIALIZACIÓN INTELIGENTE: Solo creamos la cadena RAG si vamos a chatear.
            # Esto solo crea clientes API ligeros, no carga más modelos.
            rag_chain = create_rag_chain(config)
            print("Cadena RAG creada. Lista para recibir consultas.")
            
            await chat_loop(rag_chain)

    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Asegurarnos de detener el servidor vLLM al final
        await vllm_manager.stop_server()

    print("\n" + "="*25 + " ARES-mini Test Runner Finalizado " + "="*25)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario. Adiós.")