# --- START OF FILE ares-mini/generation/chains.py ---

import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable, 
    RunnablePassthrough, 
    RunnableBranch, 
    RunnableLambda, 
    RunnableConfig
)
from qdrant_client.http import models as rest

from ..config import AresMiniConfig
# Importamos las fábricas en lugar de los componentes directamente
from ..factories import create_llm, create_embedder, create_reranker
from ..retrieval.planner import create_query_planner
from ..retrieval.transformers import RSETransformer, MMRTransformer
from ..retrieval.retriever import CustomQdrantRetriever
from ..retrieval.auto_query import create_auto_query_chain

def format_docs(docs: List[Document]) -> str:
    """Helper para formatear documentos en una única cadena de texto."""
    if not docs:
        return "No context found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_unique_documents(doc_lists: List[List[Document]]) -> List[Document]:
    """Toma una lista de listas de documentos y devuelve una lista plana con documentos únicos."""
    unique_docs = {}
    for doc_list in doc_lists:
        for doc in doc_list:
            doc_id = doc.metadata.get("chunk_id", doc.page_content)
            unique_docs[doc_id] = doc
    return list(unique_docs.values())

async def hierarchical_search_logic(input_dict: dict, retriever: CustomQdrantRetriever, config: RunnableConfig) -> List[Document]:
    """
    Define la lógica de búsqueda en dos pasos para la estrategia 'description'.
    """
    print("--- Ejecutando Estrategia de Búsqueda Jerárquica ---")
    query = input_dict["question"]
    
    description_docs = await retriever.asearch_descriptions(query)
    
    window_ids = list(set([
        doc.metadata.get("window_id") 
        for doc in description_docs 
        if doc.metadata.get("window_id")
    ]))
    
    if not window_ids:
        print("Búsqueda Jerárquica: No se encontraron documentos relevantes en la capa de descripción.")
        return []
        
    print(f"Búsqueda Jerárquica: Encontrados {len(window_ids)} documentos raíz. Buscando en sus chunks...")
    
    qdrant_filter = rest.Filter(must=[
        rest.FieldCondition(
            key="metadata.window_id",
            match=rest.MatchAny(any=window_ids)
        )
    ])
    
    return await retriever._aget_relevant_documents(
        query, 
        run_manager=config.get('callbacks'),
        filters=qdrant_filter
    )

def create_rag_chain(config: AresMiniConfig) -> Runnable:
    """
    Ensambla la cadena RAG final, modular y configurable, utilizando fábricas
    para la instanciación de componentes.
    """
    # --- 1. Creación de Componentes vía Fábricas ---
    # Las fábricas leen la configuración y devuelven la implementación correcta (vLLM o local).
    llm = create_llm(config)
    embeddings = create_embedder(config)
    # El reranker puede necesitar el wrapper del LLM para algunas implementaciones (como vLLM).
    reranker = create_reranker(config, llm) 

    # --- 2. Componentes de Recuperación y Planificación ---
    retriever = CustomQdrantRetriever(config=config, embeddings=embeddings)
    planner = create_query_planner(llm)

    # --- 3. Definir los Flujos de Recuperación ---
    auto_query_chain = create_auto_query_chain(llm)
    deep_search_flow = auto_query_chain | retriever.map() | RunnableLambda(get_unique_documents)
    hierarchical_flow = RunnableLambda(lambda x, cfg: asyncio.run(hierarchical_search_logic(x, retriever, cfg)))

    # --- 4. Crear el Enrutador de Estrategias ---
    router = RunnableBranch(
        (lambda x: x["strategy"].strategy == "description", hierarchical_flow),
        deep_search_flow,
    )

    # --- 5. Cadena de Refinamiento de Contexto (Dinámica y Configurable) ---
    def invoke_refinement_chain(input_dict: Dict[str, Any]) -> List[Document]:
        documents = input_dict["documents"]
        query = input_dict["question"]
        
        # El reranker creado por la fábrica es el primer paso
        refinement_chain: Runnable = reranker

        if config.retrieval.rse.use_rse:
            rse = RSETransformer(
                chunk_store_config=config.kv_store.model_dump(),
                max_segment_length=config.retrieval.rse.max_segment_length,
                overall_max_segments=config.retrieval.rse.overall_max_segments,
                min_segment_value=config.retrieval.rse.min_segment_value
            )
            refinement_chain = refinement_chain | rse
        
        if config.retrieval.mmr.use_mmr:
            mmr = MMRTransformer(
                embeddings=embeddings,
                k=config.retrieval.mmr.final_context_chunks,
                lambda_mult=config.retrieval.mmr.lambda_mult
            )
            refinement_chain = refinement_chain | mmr
            
        return refinement_chain.invoke(documents, config={"configurable": {"query": query}})

    # --- 6. Ensamblar la Cadena RAG Completa ---
    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        "Answer the user's question based ONLY on the following context:\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    
    full_rag_chain = (
        {"question": RunnablePassthrough()}
        .assign(strategy=planner)
        .assign(documents=router) 
        .assign(refined_docs=RunnableLambda(invoke_refinement_chain))
        .assign(context=lambda x: format_docs(x["refined_docs"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return full_rag_chain

# --- END OF FILE ares-mini/generation/chains.py ---