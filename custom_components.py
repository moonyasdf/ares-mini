# --- START OF FILE ares-mini/custom_components.py ---
import asyncio
from typing import Any, List, Optional, Dict
import numpy as np
import requests
from chonkie import RecursiveChunker, RecursiveRules
from langchain_core.documents.transformers import BaseDocumentTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from .config import LLMConfig, GenerationLLMConfig # Cambiado a GenerationLLMConfig y se mantiene LLMConfig para VLLMEmbeddings

# --- Componentes Basados en vLLM (API Remota) ---

class VLLMWrapper(BaseLLM):
    """
    Wrapper para interactuar con un servidor vLLM, con soporte para
    controlar el modo de razonamiento de Qwen3.
    """
    llm_config: GenerationLLMConfig # Usamos el Pydantic model

    @property
    def _llm_type(self) -> str:
        return "vllm_wrapper"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        # NUEVO: Aceptamos kwargs arbitrarios para pasarlos a la API
        **kwargs: Any
    ) -> str:
        api_url = f"{self.llm_config.api_base}/completions"
        
        payload = {
            "model": self.llm_config.model_name,
            "prompt": prompt,
            **kwargs
        }
        
        # --- NUEVA LÓGICA ---
        # Si se pasan chat_template_kwargs, los incluimos en el payload.
        if "chat_template_kwargs" in kwargs:
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]
        
        try:
            response = requests.post(api_url, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except requests.exceptions.RequestException as e:
            return f"[Error: Could not get response from vLLM: {e}]"
    
    # También actualizamos el método async para consistencia
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        # Aquí iría la implementación con un cliente async como httpx o aiohttp
        # Por simplicidad, envolvemos la llamada síncrona.
        return await asyncio.to_thread(self._call, prompt, stop, **kwargs)

    # --- NUEVO: Un método `generate` más completo que acepta los kwargs ---
    async def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.1,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        return_full_response: bool = False,
        chat_template_kwargs: Optional[Dict[str, Any]] = None, # Parámetro explícito
        **kwargs
    ) -> Any:
        # Este método es una simplificación para ilustrar el paso de parámetros.
        # Una implementación real usaría un cliente async y manejaría streaming.
        
        # Construimos el payload para la API de Chat (más común)
        messages = history or []
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.llm_config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            **kwargs
        }
        
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        api_url = f"{self.llm_config.api_base}/chat/completions"
        
        # Simulación de llamada asíncrona
        def sync_request():
            try:
                response = requests.post(api_url, json=payload, timeout=120)
                response.raise_for_status()
                # Para replicar la funcionalidad del reranker, necesitamos poder
                # devolver el objeto de respuesta completo.
                if return_full_response:
                    # Devolvemos un objeto compatible con el esperado por el reranker
                    from types import SimpleNamespace
                    
                    # OpenAI client returns a SimpleNamespace object, we simulate that.
                    class FakeChoice:
                        def __init__(self, data):
                            self.message = SimpleNamespace(**data.get('message', {}))
                            self.logprobs = SimpleNamespace(content=[SimpleNamespace(**lp) for lp in data.get('logprobs', {}).get('content', [])]) if data.get('logprobs') else None

                    class FakeCompletion:
                        def __init__(self, data):
                            self.choices = [FakeChoice(c) for c in data.get('choices', [])]

                    return FakeCompletion(response.json())
                
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error en VLLMWrapper.generate: {e}")
                return None

        return await asyncio.to_thread(sync_request)


class VLLMEmbeddings(Embeddings):
    """Wrapper para usar el endpoint de embeddings de un servidor vLLM."""
    llm_config: LLMConfig

    def _embed(self, texts: List[str]) -> List[List[float]]:
        api_url = f"{self.llm_config.api_base}/embeddings"
        payload = {"model": self.llm_config.embedding_model, "input": texts}
        try:
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        except requests.exceptions.RequestException as e:
            print(f"Error calling vLLM embedding API: {e}")
            # Devuelve una lista de vectores cero con la dimensión esperada si la API falla
            return [[0.0] * 1024 for _ in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# --- Componentes de Procesamiento de Documentos ---

class ChonkieTextSplitter(TextSplitter):
    """Un text splitter que utiliza la biblioteca `chonkie` para un chunking semántico."""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs: Any):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.chunker = RecursiveChunker(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            rules=RecursiveRules.from_recipe("markdown", lang="en")
        )
    def split_text(self, text: str) -> List[str]:
        if not text: return []
        chunks = self.chunker(text)
        return [chunk.text for chunk in chunks]

class AresDocumentLoader(BaseModel):
    """Cargador de documentos que soporta PyMuPDF para archivos PDF."""
    file_path: str

    def load(self) -> List[Document]:
        if self.file_path.lower().endswith(".pdf"):
            return PyMuPDFLoader(self.file_path).load()
        raise NotImplementedError(f"Loader for {self.file_path.split('.')[-1]} not implemented.")

# --- Implementaciones Locales para Qwen3 (Usando Transformers) ---

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Función de pooling específica para los modelos de embedding Qwen3."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class QwenLocalEmbedder(Embeddings):
    """Implementación local para los modelos de embedding de Qwen3."""
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        print(f"QwenLocalEmbedder cargado en {self.device}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Los documentos de recuperación no necesitan la instrucción
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # Las consultas sí necesitan la instrucción formateada
        instructed_text = f'Instruct: {self.task_description}\nQuery: {text}'
        return self._embed([instructed_text])[0]

class QwenLocalReranker(BaseDocumentTransformer):
    """Implementación local para los modelos de reranking de Qwen3."""
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", top_n: int = 5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.token_yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.token_no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        self.top_n = top_n
        print(f"QwenLocalReranker cargado en {self.device}")

    def transform_documents(self, documents: List[Document], **kwargs: Any) -> List[Document]:
        query = kwargs.get("query")
        if not query or not documents:
            return documents
        
        pairs = [[query, doc.page_content] for doc in documents]
        
        with torch.no_grad():
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            prompts = [
                f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}<|im_end|>\n<|im_start|>assistant\n"
                for q, d in pairs
            ]
            
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=4096).to(self.device)
            
            # Obtener los logits del último token
            logits = self.model(**inputs).logits[:, -1]
            
            # Extraer los logits para "yes" y "no"
            yes_logits = logits[:, self.token_yes_id]
            no_logits = logits[:, self.token_no_id]
            
            # Calcular scores con softmax
            scores = torch.nn.functional.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
            
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = score.item()

        sorted_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        return sorted_docs[:self.top_n]

    async def atransform_documents(self, documents: List[Document], **kwargs: Any) -> List[Document]:
        return await asyncio.to_thread(self.transform_documents, documents, **kwargs)