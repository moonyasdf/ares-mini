# --- ares-mini/ingestion/lsc_divider.py (VERSIÓN MEJORADA CON LLM) ---

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from ..custom_components import VLLMWrapper

# --- Pydantic models para una salida estructurada y fiable del LLM ---

class SectionDefinition(BaseModel):
    title: str = Field(description="Un título descriptivo y conciso para esta sección semántica.")
    summary: str = Field(description="Un resumen objetivo de 1-2 frases del contenido principal de la sección.")
    start_line: int = Field(description="El número de línea donde comienza esta sección.")

class SectionList(BaseModel):
    sections: List[SectionDefinition] = Field(description="Una lista de todas las secciones semánticas identificadas en el documento.")

# --- Prompt para el LLM ---

SEMANTIC_SECTIONING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un experto en estructurar documentos. Tu tarea es analizar el siguiente texto, que tiene números de línea, e identificar las principales secciones semánticas. "
         "Piensa en ello como crear un índice detallado. Para cada sección, proporciona un título, un resumen breve y la línea de inicio. "
         "Asegúrate de que las secciones cubran todo el documento de principio a fin. Responde únicamente en el formato JSON solicitado."),
        ("user",
         "Documento:\n---\n{document_with_lines}\n---\n\n"
         "Ahora, proporciona la lista de secciones en formato JSON.")
    ]
)

class LlmPoweredLSCDivider:
    """
    Usa un LLM para dividir un documento en secciones semánticas y generar un título y resumen para cada una.
    """
    def __init__(self, llm: VLLMWrapper):
        # Creamos una cadena LCEL que fuerza la salida en el formato de nuestro Pydantic model
        self.chain = SEMANTIC_SECTIONING_PROMPT | llm.with_structured_output(SectionList)
        print("LlmPoweredLSCDivider inicializado.")

    async def divide_and_enrich(self, document_text: str) -> List[Dict[str, Any]]:
        """
        Divide el texto, enriquece cada sección y devuelve una lista de diccionarios.
        Cada diccionario contiene 'title', 'summary', y 'content'.
        """
        if not document_text.strip():
            return []

        lines = document_text.split('\n')
        # Añadir números de línea para que el LLM pueda referenciarlos
        text_with_lines = "\n".join(f"[{i+1}]: {line}" for i, line in enumerate(lines))

        print("LlmPoweredLSCDivider: Invocando al LLM para identificar secciones semánticas...")
        try:
            section_list_result = await self.chain.ainvoke({"document_with_lines": text_with_lines})
        except Exception as e:
            print(f"Error al invocar la cadena de división LSC: {e}. Se tratará el documento como una sola sección.")
            section_list_result = None

        if not section_list_result or not section_list_result.sections:
            print("LlmPoweredLSCDivider: El LLM no devolvió secciones válidas. Usando el documento completo como fallback.")
            return [{"title": "Documento Completo", "summary": None, "content": document_text}]

        # Ordenar las secciones por su línea de inicio para asegurar el orden correcto
        sections = sorted(section_list_result.sections, key=lambda s: s.start_line)
        
        # Reconstruir el contenido de cada sección a partir de los límites de línea
        divided_sections = []
        for i, section in enumerate(sections):
            start_line_idx = section.start_line - 1
            
            # El fin de la sección es el inicio de la siguiente, o el final del documento.
            end_line_idx = sections[i+1].start_line - 1 if i + 1 < len(sections) else len(lines)
            
            # Asegurarse de que los índices son válidos
            if start_line_idx >= len(lines) or start_line_idx < 0:
                continue
            
            section_content = "\n".join(lines[start_line_idx:end_line_idx])
            
            divided_sections.append({
                "title": section.title,
                "summary": section.summary,
                "content": section_content.strip()
            })
            
        print(f"LlmPoweredLSCDivider: Documento dividido en {len(divided_sections)} secciones semánticas.")
        return divided_sections