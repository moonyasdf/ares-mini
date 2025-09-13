# --- START OF FILE ares-mini/ingestion/description_generator.py ---

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from ..custom_components import VLLMWrapper

class DocumentDescription(BaseModel):
    """Descripción estructurada de un documento."""
    title: str = Field(description="Un título conciso y descriptivo para el documento.")
    summary: str = Field(description="Un resumen objetivo de 1-2 frases del contenido principal.")

def create_description_generator(llm: VLLMWrapper) -> Runnable:
    """Crea una cadena que genera una descripción estructurada de un texto."""
    DESC_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un experto en resumir y titular textos. Extrae un título y un resumen conciso del siguiente documento. Responde únicamente en el formato JSON solicitado."),
            ("user", "{text}")
        ]
    )
    return DESC_PROMPT | llm.with_structured_output(DocumentDescription)