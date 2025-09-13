"""
Query Routing Logic for Ares-Mini.

This module implements a dynamic query router that intelligently selects the
most appropriate retrieval strategy based on the user's query. This is a crucial
component for achieving high-quality retrieval, as different types of queries
benefit from different retrieval methods.

For example, a question with specific keywords might be best served by a sparse
retriever like BM25, while a more conceptual question would benefit from dense,
semantic retrieval. This router uses a language model to classify the query's
intent and then dispatches it to the corresponding retriever chain.

This replaces the static logic of the original `selector.py` with a more
flexible and powerful LLM-driven approach.
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableLambda

from ..custom_components import VLLMWrapper

# --- Pydantic Model for Router Output ---

class RouteQuery(BaseModel):
    """Defines the output schema for the query routing decision."""
    retrieval_strategy: Literal["vector", "keyword", "hybrid"] = Field(
        ...,
        description="The optimal retrieval strategy to use for the given query."
    )

# --- Router Chain Construction ---

def create_router_chain(llm: VLLMWrapper) -> Runnable:
    """
    Creates a chain that classifies a user's query and decides on a retrieval strategy.

    Args:
        llm (VLLMWrapper): The vLLM-backed language model to use for the routing logic.

    Returns:
        Runnable: A chain that takes a user query and returns a RouteQuery object.
    """
    # This prompt guides the LLM to act as a query router.
    # It's a key part of making the retrieval process intelligent.
    ROUTER_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at analyzing user queries and routing them to the best retrieval strategy. "
         "Based on the query, classify it as either 'vector' for semantic questions, 'keyword' for specific term searches, or 'hybrid' for mixed queries."),
        ("user", "{question}")
    ])

    # The structured_llm will be forced to output in the format of the RouteQuery model.
    # This ensures reliable, parsable output from the LLM.
    structured_llm = llm.with_structured_output(RouteQuery)

    router_chain = ROUTER_PROMPT | structured_llm
    return router_chain

# --- Example of how to use the router with branching ---

def get_full_retrieval_chain(llm: VLLMWrapper, vector_retriever, hybrid_retriever, keyword_retriever) -> Runnable:
    """
    Assembles the complete retrieval chain, including the router and its branches.

    This function demonstrates how to use the router's output to dynamically select
    which retriever to execute. This is a powerful pattern in LCEL.

    Args:
        llm (VLLMWrapper): The language model.
        vector_retriever: The retriever for semantic search.
        hybrid_retriever: The retriever for hybrid search.
        keyword_retriever: The retriever for keyword search.

    Returns:
        A runnable that takes a question and returns the retrieved documents.
    """
    router_chain = create_router_chain(llm)

    def select_retriever(route: RouteQuery):
        """A function to select the retriever based on the router's decision."""
        if route.retrieval_strategy == "vector":
            return vector_retriever
        if route.retrieval_strategy == "hybrid":
            return hybrid_retriever
        return keyword_retriever

    # The final chain: the input question is passed to the router.
    # The router's output (a RouteQuery object) is then used by `select_retriever`
    # to pick the correct path to execute.
    # The original question is then passed to the selected retriever.
    # This is a placeholder for the final implementation in generation/chains.py
    full_chain = router_chain | RunnableLambda(select_retriever)
    return full_chain
