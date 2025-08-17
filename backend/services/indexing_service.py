import os
import hashlib
import json
import gc # Importamos el recolector de basura
from typing import List, Dict, Any, Generator, Tuple
from pypdf import PdfReader
from core.vectorstore import VectorStore
from core.embeddings import Embeddings

class IndexingService:
    def __init__(self, vs: VectorStore, emb: Embeddings, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.vs = vs
        self.emb = emb
        self.chunk_size = chunk_size
        # BUG CORREGIDO: Usar el parámetro correcto para el solapamiento.
        self.chunk_overlap = chunk_overlap
        
    def _chunk_text(self, text: str) -> List[str]:
        """Divide el texto en fragmentos con un solapamiento definido."""
        if not text:
            return []
        
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            # Lógica de avance corregida para una ventana deslizante.
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sha256(self, path: str) -> str:
        """Calcula el hash SHA256 de un archivo de forma eficiente."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    
    # NUEVO MÉTODO GENERADOR: Procesa el PDF página por página sin cargarlo todo en memoria.
    def _iter_pdf_chunks(self, pdf_path: str) -> Generator[Tuple[str, Dict], None, None]:
        """
        Generador que lee un PDF y produce tuplas de (texto_del_chunk, metadata).
        Esto evita cargar todo el archivo en la memoria RAM.
        """
        basename = os.path.basename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 40: # Filtra páginas vacías o con poco contenido.
                    continue
                
                chunks = self._chunk_text(text)
                for j, chunk_text in enumerate(chunks):
                    metadata = {"source": basename, "page": i + 1, "chunk": j}
                    yield chunk_text, metadata
        except Exception as e:
            print(f"Error al procesar el PDF {pdf_path}: {e}")
            return

    # MÉTODO PRINCIPAL MODIFICADO: Ahora usa lotes (batches).
    def index_pdf(self, pdf_path: str, force: bool = False, batch_size: int = 16) -> int:
        """
        Indexa un PDF procesándolo en lotes para mantener un bajo consumo de memoria.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No existe el PDF en {pdf_path}")

        state_path = pdf_path + ".index.json"
        current_hash = self._sha256(pdf_path)

        if not force:
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("sha256") == current_hash:
                    print(f"El archivo '{os.path.basename(pdf_path)}' ya está indexado y no ha cambiado. Omitiendo.")
                    return int(state.get("chunks", 0))
            except Exception:
                pass

        # Si forzamos la reindexación, reiniciamos el vector store.
        print(f"Iniciando indexación para '{os.path.basename(pdf_path)}'...")
        self.vs.reset()

        texts_batch: List[str] = []
        metas_batch: List[Dict[str, Any]] = []
        total_chunks = 0

        # Función auxiliar para procesar y limpiar un lote.
        def flush_batch():
            nonlocal total_chunks
            if not texts_batch:
                return

            print(f"Procesando lote de {len(texts_batch)} chunks...")
            embs = self.emb.embed(texts_batch)
            self.vs.add_texts(texts=texts_batch, metadatas=metas_batch, embeddings=embs)
            
            total_chunks += len(texts_batch)
            texts_batch.clear()
            metas_batch.clear()
            gc.collect() # Sugiere al recolector de basura que libere memoria.

        # Bucle principal que consume del generador.
        for text, meta in self._iter_pdf_chunks(pdf_path):
            texts_batch.append(text)
            metas_batch.append(meta)

            if len(texts_batch) >= batch_size:
                flush_batch()
        
        # Procesar el último lote si queda algo.
        flush_batch()

        # Guardar el estado final.
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"sha256": current_hash, "chunks": total_chunks}, f)

        print(f"\n✅ Indexación completa. Total de {total_chunks} chunks guardados.")
        return total_chunks