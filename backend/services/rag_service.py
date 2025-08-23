from typing import List, Dict, Any
from core.embeddings import Embeddings
from core.llm import LLM
from core.vectorstore import VectorStore

class RAGService:
    def __init__(self, vs: VectorStore, emb: Embeddings, llm: LLM):
        """Retrieval-Augmented Generation (RAG) service.

        This class provides an interface that connects a vector store, 
        embeddings model, and a language model to enable RAG-based querying.

        Args:
            vs (VectorStore): Vector store instance used for similarity search.
            emb (Embeddings): Embedding model for encoding queries.
            llm (LLM): Language model used to generate answers.
        """
        self.vs = vs
        self.emb = emb
        self.llm = llm
        
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """Build the prompt for the LLM using the retrieved contexts.

        Args:
            question (str): The user question.
            contexts (List[str]): List of text passages retrieved from the vector store.

        Returns:
            str: A formatted prompt containing the context and the user question.
        """
        context_block = "\n\n---\n\n".join(contexts)
        return (
            "Eres un asistente que responde basándote EXCLUSIVAMENTE en el contexto de la tesis. "
            "Si la respuesta no está en el contexto, di que no aparece en el documento. "
            "Cita página(s) cuando sea posible.\n\n"
            f"Contexto:\n{context_block}\n\n"
            f"Pregunta: {question}\n\n"
            "Respuesta concisa y bien estructurada:"
        )

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Query the RAG pipeline to answer a question based on the thesis.

        Steps:
            1. Embed the input question.
            2. Retrieve the top-k most relevant contexts from the vector store.
            3. Build a prompt with the retrieved contexts and the question.
            4. Generate an answer using the language model.

        Args:
            question (str): The input question to be answered.
            k (int, optional): Number of contexts to retrieve. Defaults to 4.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "answer" (str): Generated answer from the LLM.
                - "contexts" (List[Dict[str, Any]]): Retrieved contexts with text, metadata, and distance.
        """
        q_emb = self.emb.embed([question])[0]
        res = self.vs.query(q_emb, k=k)

        contexts = []
        for txt, meta, dist in zip(res["documents"], res["metadatas"], res["distances"]):
            contexts.append({"text": txt, "metadata": meta, "distance": float(dist)})
        prompt = self._build_prompt(question, [c["text"] for c in contexts])
        
        answer = self.llm.generate(prompt)
        return {"answer": answer, "contexts": contexts}