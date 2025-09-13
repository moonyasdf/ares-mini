# --- START OF FILE ares-mini/retrieval/auto_query.py ---

from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnable import Runnable

from ..custom_components import VLLMWrapper

# El prompt se inspira en las mejores prácticas de los tutoriales de LangChain.
# Es claro, directo y le pide al LLM que se centre en la diversidad de perspectivas.
MULTI_QUERY_PROMPT_TEMPLATE = """
You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help 
the user overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines. Do not number them.

Original question: {question}
"""

def create_auto_query_chain(llm: VLLMWrapper) -> Runnable:
    """
    Crea una cadena LCEL que toma una pregunta y devuelve una lista de sub-preguntas.

    Esta cadena incluye la pregunta original en la lista final para asegurar
    que la intención principal del usuario nunca se pierda.

    Args:
        llm (VLLMWrapper): El modelo de lenguaje para la generación de consultas.

    Returns:
        Runnable: Una cadena que transforma una pregunta (str) en una lista de preguntas (List[str]).
    """
    
    # 1. Definir el prompt para la generación de consultas.
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)

    # 2. Definir la cadena de generación.
    # El `StrOutputParser` asegura que la salida sea una cadena de texto limpia.
    generate_queries_chain = prompt | llm | StrOutputParser()

    # 3. Definir una función para parsear la salida y añadir la pregunta original.
    def parse_and_add_original(response: str, original_question: str) -> List[str]:
        """
        Toma la salida del LLM (una cadena con saltos de línea), la divide en una lista,
        y añade la pregunta original al principio.
        """
        # Divide las preguntas generadas y filtra las líneas vacías.
        generated_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        # Devuelve la pregunta original más las generadas.
        all_queries = [original_question] + generated_queries
        
        print(f"--- AutoQuery: Generadas {len(all_queries)} consultas para expandir la búsqueda. ---")
        return all_queries

    # 4. Ensamblar la cadena final.
    # Usamos un diccionario para pasar la pregunta original a través de la cadena.
    auto_query_chain = (
        {"question": lambda x: x} # Mantiene la pregunta original
        | {
            "response": generate_queries_chain, # Genera nuevas preguntas
            "original_question": lambda x: x["question"]
          }
        | (lambda x: parse_and_add_original(x["response"], x["original_question"]))
    )
    
    return auto_query_chain