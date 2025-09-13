# --- START OF FILE ares-mini/retrieval/planner.py ---

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable

from ..custom_components import VLLMWrapper

class RetrievalStrategy(BaseModel):
    """Define la estrategia de recuperación óptima para una consulta."""
    strategy: Literal["vector", "hybrid", "description"] = Field(
        ...,
        description="La estrategia de recuperación a usar: 'vector' para semántica, 'hybrid' para mezcla, 'description' para resúmenes."
    )
    reasoning: str = Field(..., description="Breve razonamiento de la elección.")

def create_query_planner(llm: VLLMWrapper) -> Runnable:
    """Crea una cadena que clasifica la consulta de un usuario y decide una estrategia."""
    ROUTER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Eres un planificador experto para un sistema RAG. Tu tarea es analizar la consulta del usuario y "
             "elegir la mejor estrategia de recuperación. Opciones:\n"
             "- 'description': Para preguntas generales, de alto nivel o que buscan un resumen de un documento.\n"
             "- 'vector': Para preguntas semánticas muy específicas sobre un detalle concreto.\n"
             "- 'hybrid': Para la mayoría de las preguntas que mezclan palabras clave con intención semántica."),
            ("user", "Analiza la siguiente pregunta y elige la mejor estrategia:\n\n> {question}")
        ]
    )
    structured_llm = llm.with_structured_output(RetrievalStrategy)
    return ROUTER_PROMPT | structured_llm