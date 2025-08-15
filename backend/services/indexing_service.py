import os, hashlib, json
from typing import List, Dict, Any
from pypdf import PdfReader
from core.vectorstore import VectorStore
from core.embeddings import Embeddings

class IndexingService:
    def __init__(self, vs: VectorStore, emb: Embeddings, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.vs = vs
        self.emb = emb
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_size
        
    def _chunk_text(self, text: str) -> List[str]:
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _extract_pdf_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages.append({
                "page": i + 1,
                "text": txt
            })
        return pages
    
    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def index_pdf(self, pdf_path: str, force: bool = False) -> int:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No existe el PDF en {pdf_path}")

        state_path = pdf_path + ".index.json"
        current_hash = self._sha256(pdf_path)

        if not force:
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("sha256") == current_hash:
                    # Ya indexado y sin cambios
                    return int(state.get("chunks", 0))
            except Exception:
                pass

        # Reindexación completa
        self.vs.reset()
        pages = self._extract_pdf_text(pdf_path)

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        basename = os.path.basename(pdf_path)

        for p in pages:
            for j, ch in enumerate(self._chunk_text(p["text"])):
                texts.append(ch)
                metas.append({"source": basename, "page": p["page"], "chunk": j})

        if not texts:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"sha256": current_hash, "chunks": 0}, f)
            return 0

        embs = self.emb.embed(texts)
        self.vs.add_texts(texts=texts, metadatas=metas, embeddings=embs)

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"sha256": current_hash, "chunks": len(texts)}, f)

        return len(texts)