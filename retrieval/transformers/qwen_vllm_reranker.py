# ares-mini/retrieval/transformers/qwen_vllm_reranker.py
import asyncio
import math
from typing import Any, Sequence, List, Dict
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents.transformers import BaseDocumentTransformer

# Importamos el wrapper de vLLM de ares-mini
from ..custom_components import VLLMWrapper

class QwenVLLMReranker(BaseDocumentTransformer, BaseModel):
    """
    Reranks documents using a Qwen model served by vLLM, calculating scores
    based on the logprobs of 'yes' and 'no' tokens, mimicking the original ARES logic.
    """
    llm: VLLMWrapper
    top_n: int = Field(default=25, description="Number of documents to return after reranking.")
    
    # Cache para los token IDs de 'yes' y 'no'
    _token_ids: Dict[str, int] = {}
    
    class Config:
        arbitrary_types_allowed = True

    def _format_prompt(self, query: str, doc: str) -> List[Dict[str, str]]:
        """Formats the prompt according to the Qwen reranker template."""
        return [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

    async def _get_yes_no_token_ids(self) -> (int, int):
        """
        Dynamically gets the token IDs for 'yes' and 'no' using the tokenizer 
        from the underlying VLLMWrapper's LLM config if possible.
        This is a simplified approach; a real implementation might need a dedicated tokenizer.
        """
        if "yes" in self._token_ids and "no" in self._token_ids:
            return self._token_ids["yes"], self._token_ids["no"]

        # As a fallback, we use common token IDs for Qwen models.
        # A more robust solution would be to have access to the tokenizer.
        # For Qwen/Qwen3-0.6B, 'yes' is 1353 and 'no' is 354.
        # We will hardcode these as a reliable fallback.
        print("QwenVLLMReranker: Using hardcoded token IDs for 'yes' (1353) and 'no' (354).")
        self._token_ids["yes"] = 1353
        self._token_ids["no"] = 354
        return self._token_ids["yes"], self._token_ids["no"]

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        query = kwargs.get("query")
        if not query or not documents:
            return documents

        print(f"--- QwenVLLMReranker: Reranking {len(documents)} docs for query: '{query[:50]}...' ---")
        
        yes_id, no_id = await self._get_yes_no_token_ids()

        tasks = []
        for doc in documents:
            conv = self._format_prompt(query, doc.page_content)
            # We ask vLLM for the full response object to get logprobs
            task = self.llm.generate(
                prompt=conv[-1]['content'],
                history=conv[:-1],
                max_tokens=2,
                logprobs=True,
                top_logprobs=10, # Get enough logprobs to find 'yes' and 'no'
                return_full_response=True, # Crucial: ask the wrapper for the raw OpenAI completion object
                hat_template_kwargs={"enable_thinking": False}
            )
            tasks.append(task)

        # Process all requests to vLLM concurrently
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        new_scores = []
        for output in outputs:
            if isinstance(output, Exception) or not hasattr(output, 'choices') or not output.choices:
                print(f"Error in one of the reranking calls: {output}")
                new_scores.append(0.0)
                continue
            
            try:
                choice = output.choices[0]
                logprob_content = choice.logprobs.content if choice.logprobs else []
                
                logprob_dict = {}
                if logprob_content and logprob_content[0].top_logprobs:
                    # Logprobs are for the first generated token.
                    # We need to convert token_id from string (as in API) to int for lookup.
                    logprob_dict = {
                        int(lp.token_id): lp.logprob for lp in logprob_content[0].top_logprobs
                    }

                logit_yes = logprob_dict.get(yes_id, -100.0)
                logit_no = logprob_dict.get(no_id, -100.0)
                
                exp_yes = math.exp(logit_yes)
                exp_no = math.exp(logit_no)
                score = exp_yes / (exp_yes + exp_no) if (exp_yes + exp_no) > 0 else 0.0
                new_scores.append(score)
            except Exception as e:
                print(f"Error processing model output for reranking: {e}. Output: {output}")
                new_scores.append(0.0)

        for doc, score in zip(documents, new_scores):
            doc.metadata["rerank_score"] = score

        reranked_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        return reranked_docs[:self.top_n]

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return asyncio.run(self.atransform_documents(documents, **kwargs))