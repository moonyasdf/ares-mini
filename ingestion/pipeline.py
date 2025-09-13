# --- ares-mini/ingestion/pipeline.py (CÓDIGO COMPLETO Y ACTUALIZADO) ---

import asyncio
import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from tqdm import tqdm
import tiktoken

from ..config import AresMiniConfig
from ..custom_components import VLLMWrapper, AresDocumentLoader, ChonkieTextSplitter, VLLMEmbeddings
from ..storage.vector_db import get_vector_store
from ..storage.kv_store import get_kv_store
from .description_generator import create_description_generator, DocumentDescription
# --- NUEVA IMPORTACIÓN ---
from .lsc_divider import LlmPoweredLSCDivider
from .embedder import LateChunkingEmbedder
from .embedder_sparse import FastEmbedSparseEmbedder

class IngestionOrchestrator:
    """
    Orquesta la ingesta masiva de documentos, con soporte para Semantic Sectioning.
    """
    def __init__(self, config: AresMiniConfig):
        self.config = config
        self.llm = VLLMWrapper(llm_config=config.generation_llm)
        self.sparse_embedder = FastEmbedSparseEmbedder(model_name=config.sparse_embedder.model_name)
        self.vector_store = get_vector_store(config.vector_store, embeddings=VLLMEmbeddings(llm_config=config.generation_llm))
        self.kv_store = get_kv_store(config.kv_store)
        self.description_generator = create_description_generator(self.llm)
        # --- NUEVO: Instanciamos el divisor LSC impulsado por LLM ---
        self.lsc_divider = LlmPoweredLSCDivider(self.llm)
        self.late_chunking_embedder = LateChunkingEmbedder(vllm_wrapper=self.llm, config=config)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            print("Advertencia: Tiktoken no pudo cargarse.")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    # --- NUEVO: Helper para construir headers jerárquicos ---
    def _build_autocontent_header(self, doc_description: Optional[DocumentDescription], section: Dict[str, Any]) -> str:
        """Construye el header de AutoContext usando info del documento y de la sección."""
        parts = []
        if doc_description:
            if doc_description.title:
                parts.append(f"Título del Documento: {doc_description.title}")
            if doc_description.summary:
                parts.append(f"Resumen del Documento: {doc_description.summary}")
        
        # Añadir información de la sección semántica
        if section.get("title") and section["title"] != "Documento Completo":
             parts.append(f"Título de la Sección: {section['title']}")
        if section.get("summary"):
             parts.append(f"Resumen de la Sección: {section['summary']}")
        
        if not parts:
            return ""
        
        return "Contexto General: " + ". ".join(parts) + ". Contenido Específico del Fragmento: "

    async def _ingest_content_async(self, identifier: str, content: str, metadata: Dict[str, Any], description: Any = None):
        """Procesa un bloque de contenido (archivo o meta-ventana)."""
        print(f"Procesando: {identifier} (Tamaño: {len(content)} chars)")
        
        # --- NUEVA LÓGICA DE DIVISIÓN SEMÁNTICA ---
        # 1. Dividir el contenido en secciones semánticas enriquecidas
        enriched_sections = await self.lsc_divider.divide_and_enrich(content)

        all_chunks_docs = []
        all_texts_for_embedding = []

        window_id = f"win_{hash(content[:500])}"

        # 2. Iterar sobre cada sección semántica
        for section in enriched_sections:
            section_content = section["content"]
            if not section_content:
                continue

            # 3. Construir el header jerárquico para esta sección
            header = self._build_autocontent_header(description, section)

            # 4. Chunking de la sección
            splitter = ChonkieTextSplitter(
                chunk_size=self.config.ingestion.chunk_size,
                chunk_overlap=self.config.ingestion.chunk_overlap
            )
            chunk_texts_on_section = splitter.split_text(section_content)
            
            for i, text in enumerate(chunk_texts_on_section):
                chunk_counter = len(all_chunks_docs)
                chunk_metadata = {
                    **metadata,
                    "window_id": window_id,
                    "section_title": section.get("title"), # Añadimos metadatos de sección
                    "chunk_in_section_index": i,
                    "chunk_id": f"{window_id}_chunk_{chunk_counter}"
                }
                all_chunks_docs.append(Document(page_content=text, metadata=chunk_metadata))
                
                # Preparamos el texto con header para el embedding
                all_texts_for_embedding.append(header + text)
        
        if not all_chunks_docs:
            print(f"Advertencia: No se generaron chunks para {identifier}")
            return

        chunk_texts = [doc.page_content for doc in all_chunks_docs]

        print(f"Generando embeddings híbridos para {len(chunk_texts)} chunks...")
        dense_task = self.late_chunking_embedder.embed_document_with_late_chunking(content, headers=[self._build_autocontent_header(description, {})])
        sparse_task = self.sparse_embedder.aembed_documents(chunk_texts)
        dense_embeddings_map, sparse_embeddings_list = await asyncio.gather(dense_task, sparse_task)

        documents_for_vdb = []
        vectors_for_vdb = []
        kv_items_to_set = []

        for i, doc in enumerate(all_chunks_docs):
            text = doc.page_content
            if text in dense_embeddings_map and i < len(sparse_embeddings_list) and sparse_embeddings_list[i]:
                kv_items_to_set.append((doc.metadata["chunk_id"], {"text": text, "metadata": doc.metadata}))
                documents_for_vdb.append(doc)
                vectors_for_vdb.append({
                    "dense": dense_embeddings_map[text],
                    "sparse": sparse_embeddings_list[i]
                })

        if documents_for_vdb:
            await asyncio.to_thread(self.vector_store.add_vectors, documents=documents_for_vdb, vectors=vectors_for_vdb)

        if description:
            desc_text_for_embedding = f"Title: {description.title}\nSummary: {description.summary}"
            desc_embedding = VLLMEmbeddings(llm_config=self.llm.llm_config).embed_query(desc_text_for_embedding)
            desc_doc = Document(
                page_content=desc_text_for_embedding,
                metadata={"is_description": True, "window_id": window_id, "source_files": metadata.get("source_files", [])}
            )
            desc_vectors = [{"dense": desc_embedding, "sparse": {"indices": [], "values": []}}]
            await asyncio.to_thread(self.vector_store.add_vectors, documents=[desc_doc], vectors=desc_vectors)
            kv_items_to_set.append((f"desc_{window_id}", description.model_dump()))

        chunk_ids = [d.metadata["chunk_id"] for d in documents_for_vdb]
        kv_items_to_set.append((f"win_meta_{window_id}", {"window_id": window_id, "chunk_ids": chunk_ids}))
        
        if kv_items_to_set:
            await asyncio.to_thread(self.kv_store.mset, kv_items_to_set)

    async def ingest_directory_as_meta_windows(self, directory_path: str, max_window_tokens: int = 7000):
        """
        Ingesta un directorio completo agrupando archivos en 'meta-ventanas' para
        minimizar las llamadas al LLM de descripción y aplicar división semántica.
        """
        supported_exts = (".pdf", ".txt", ".md")
        filepaths = [os.path.join(root, file) for root, _, files in os.walk(directory_path) for file in files if file.endswith(supported_exts)]
        
        if not filepaths:
            print(f"No se encontraron documentos soportados en {directory_path}")
            return

        print(f"Se encontraron {len(filepaths)} documentos. Agrupando en meta-ventanas de ~{max_window_tokens} tokens.")
        
        meta_windows = []
        current_window_text = ""
        current_window_files = []
        current_token_count = 0

        for fp in tqdm(filepaths, desc="Agrupando archivos en meta-ventanas"):
            try:
                loaded_pages = AresDocumentLoader(file_path=fp).load()
                file_content = "\n\n".join([doc.page_content for doc in loaded_pages])
                file_token_count = self._count_tokens(file_content)

                if current_token_count + file_token_count > max_window_tokens and current_window_text:
                    meta_windows.append({
                        "content": current_window_text,
                        "source_files": current_window_files,
                    })
                    current_window_text = file_content
                    current_window_files = [fp]
                    current_token_count = file_token_count
                else:
                    current_window_text += "\n\n" + file_content
                    current_window_files.append(fp)
                    current_token_count += file_token_count
            except Exception as e:
                print(f"Error cargando el archivo {fp}, se omitirá. Error: {e}")

        if current_window_text:
            meta_windows.append({
                "content": current_window_text,
                "source_files": current_window_files,
            })

        print(f"Se crearon {len(meta_windows)} meta-ventanas para procesar.")
        
        descriptions = {}
        if self.config.ingestion.generate_descriptions:
            print("Generando descripciones para todas las meta-ventanas en un lote masivo...")
            prompts = [{"text": mw["content"][:8000]} for mw in meta_windows]
            desc_results = await self.description_generator.abatch(prompts)
            for i, desc in enumerate(desc_results):
                descriptions[hash(meta_windows[i]["content"])] = desc
        
        ingest_tasks = []
        for i, mw in enumerate(meta_windows):
            window_hash = hash(mw["content"])
            description = descriptions.get(window_hash)
            metadata = {"source_files": mw["source_files"]}
            representative_name = os.path.basename(mw["source_files"][0]) + (f" (+{len(mw['source_files'])-1} más)" if len(mw['source_files']) > 1 else "")
            
            task = self._ingest_content_async(
                identifier=f"meta-window-{i}_{representative_name}",
                content=mw["content"],
                metadata=metadata,
                description=description
            )
            ingest_tasks.append(task)
        
        await tqdm.gather(*ingest_tasks, desc="Ingestando meta-ventanas")
        print(f"--- Ingesta masiva optimizada completada para el directorio {directory_path} ---")


def create_ingestion_pipeline(config: AresMiniConfig) -> Runnable:
    orchestrator = IngestionOrchestrator(config)

    def route(input_path: str):
        loop = asyncio.get_event_loop()
        if os.path.isdir(input_path):
            loop.run_until_complete(orchestrator.ingest_directory_as_meta_windows(input_path))
        elif os.path.isfile(input_path):
            async def run_single():
                desc = None
                content = "\n\n".join([doc.page_content for doc in AresDocumentLoader(file_path=input_path).load()])
                if config.ingestion.generate_descriptions:
                    try:
                        desc = await orchestrator.description_generator.ainvoke({"text": content[:8000]})
                    except Exception as e:
                        print(f"No se pudo generar descripción para {input_path}: {e}")
                # Llamamos al nuevo método unificado
                await orchestrator._ingest_content_async(input_path, content, {"source_files": [input_path]}, description=desc)
            
            loop.run_until_complete(run_single())
        else:
            raise ValueError(f"La ruta no es un archivo o directorio válido: {input_path}")

    return RunnableLambda(route)