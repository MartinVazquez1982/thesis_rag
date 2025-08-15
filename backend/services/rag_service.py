from typing import List, Dict, Any
from core.embeddings import Embeddings
from core.llm import LLM
from core.vectorstore import VectorStore

class RAGService:
    def __init__(self, vs: VectorStore, emb: Embeddings, llm: LLM):
        self.vs = vs
        self.emb = emb
        self.llm = llm
        
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
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
        q_emb = self.emb.embed([question])[0]
        res = self.vs.query(q_emb, k=k)

        contexts = []
        for txt, meta, dist in zip(res["documents"], res["metadatas"], res["distances"]):
            contexts.append({"text": txt, "metadata": meta, "distance": float(dist)})

        prompt = self._build_prompt(question, [c["text"] for c in contexts])
        answer = self.llm.generate(prompt)
        return {"answer": answer, "contexts": contexts}