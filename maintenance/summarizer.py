# --- START OF FILE ares-mini/maintenance/summarizer.py ---

from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..custom_components import VLLMWrapper
from ..storage.kv_store import get_kv_store
from ..config import AresMiniConfig

class Summarizer:
    """
    Crea resúmenes periódicos (diarios, semanales) de las descripciones
    de los documentos ingeridos.
    """
    def __init__(self, config: AresMiniConfig):
        self.config = config
        self.llm = VLLMWrapper(llm_config=config.llm)
        self.kv_store = get_kv_store(config.kv_store)

    def _get_descriptions_for_period(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """
        Recupera descripciones de documentos del KV store dentro de un rango de fechas.
        NOTA: Esto asume que el KV store permite algún tipo de iteración o consulta.
        Una implementación simple podría iterar todas las claves.
        """
        # Esta es una simulación. Un KV store real necesitaría un método de consulta.
        print(f"Simulando búsqueda de descripciones entre {start_dt} y {end_dt}")
        # all_keys = self.kv_store.yield_keys() # Suponiendo que existe
        # ... lógica para filtrar claves por fecha ...
        return [
            {"title": "Informe Anual Nike 2023", "summary": "Resultados financieros y operativos de Nike para el año fiscal 2023."},
            {"title": "Paper sobre Agentes LLM", "summary": "Un análisis sobre los patrones de diseño para agentes autónomos."}
        ]

    def _create_summary_chain(self) -> Runnable:
        """Crea la cadena LCEL para generar el resumen consolidado."""
        SUMMARY_PROMPT = ChatPromptTemplate.from_template(
            "Eres un analista experto. A continuación se presentan los títulos y resúmenes de varios documentos procesados el día {date}. "
            "Tu tarea es crear un único párrafo que sintetice los temas más importantes del día.\n\n"
            "Documentos del día:\n{formatted_descriptions}\n\n"
            "Resumen consolidado del día:"
        )
        return SUMMARY_PROMPT | self.llm | StrOutputParser()

    def run_daily_summary(self, target_date: datetime = None):
        """Genera y almacena el resumen para un día específico."""
        if target_date is None:
            target_date = datetime.now(timezone.utc) - timedelta(days=1) # Resumen de ayer
        
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"--- Generando resumen de mantenimiento para la fecha: {date_str} ---")

        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        descriptions = self._get_descriptions_for_period(start_of_day, end_of_day)
        
        if not descriptions:
            print("No se encontraron documentos para resumir en esta fecha.")
            return

        formatted_descriptions = "\n".join(
            f"- {d['title']}: {d['summary']}" for d in descriptions
        )
        
        summary_chain = self._create_summary_chain()
        daily_summary = summary_chain.invoke({
            "date": date_str,
            "formatted_descriptions": formatted_descriptions
        })

        # Almacenar el resumen en el KV store con una clave especial
        summary_key = f"summary_daily_{date_str}"
        self.kv_store.mset([(summary_key, {"summary": daily_summary, "date": date_str})])
        
        print(f"Resumen diario guardado con éxito:\n{daily_summary}")

# Ejemplo de cómo se ejecutaría (ej. desde un script de mantenimiento)
if __name__ == '__main__':
    from ..config import load_config
    # Cargar configuración
    config = load_config()
    # Crear instancia del summarizer
    summarizer = Summarizer(config=config)
    # Ejecutar para la fecha de ayer
    summarizer.run_daily_summary()