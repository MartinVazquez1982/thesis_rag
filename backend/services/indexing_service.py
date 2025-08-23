import os
import hashlib
import json
import gc
from typing import List, Dict, Any, Generator, Tuple
from pypdf import PdfReader
from core.vectorstore import VectorStore
from core.embeddings import Embeddings

class IndexingService:
    def __init__(self, vs: VectorStore, emb: Embeddings, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        This class aims to read PDF files, process their content efficiently and save it to a vector database.
        
        Args:
            vs (VectorStore): Vector Database
            emb (Embeddings): Embedder
            chunk_size (int, optional): The maximum size of each text fragment. Defaults to 1000.
            chunk_overlap (int, optional): number of characters from the end of a fragment that are repeated at the beginning of the next. Defaults to 150.
        """
        self.vs = vs
        self.emb = emb
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Takes a long block of text and splits it into a list of chunks based on the size and overlap defined in the constructor.
        Uses a sliding window.

        Args:
            text (str): text for applies the split

        Returns:
            List[str]: chunks
        """
        if not text:
            return []
        
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sha256(self, path: str) -> str:
        """
        Calculates a unique "fingerprint" (a SHA256 hash) for the PDF file

        Args:
            path (str): file path that will apply the hash

        Returns:
            str: fingerprint (SHA256 hash)
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def _iter_pdf_chunks(self, pdf_path: str) -> Generator[Tuple[str, Dict], None, None]:
        """       
        It reads the PDF page by page, extracts the text, divides it into chunks, and "produces" them one by one.

        Args:
            pdf_path (str): PDF file path

        Yields:
            Generator[Tuple[str, Dict], None, None]: Tuple with chunk and its metadata
        """
        basename = os.path.basename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # It filter the empty pages or with little content
                if len(text.strip()) < 40:
                    continue
                
                chunks = self._chunk_text(text)
                for j, chunk_text in enumerate(chunks):
                    metadata = {"source": basename, "page": i + 1, "chunk": j}
                    yield chunk_text, metadata
        except Exception as e:
            print(f"Error al procesar el PDF {pdf_path}: {e}")
            return

    def index_pdf(self, pdf_path: str, force: bool = False, batch_size: int = 16) -> int:
        """
        Indexes the content of a PDF file by processing it in memory-efficient batches.

        This method reads a PDF, splits its text into chunks, generates embeddings
        for these chunks, and adds them to a vector store. It is designed to handle
        large files by processing the document page by page and using batches to
        avoid high memory consumption.

        The method also implements a caching mechanism. It calculates the SHA256 hash
        of the file and saves it to a `.index.json` state file upon successful
        indexing. If the method is called again on the same file and the hash has not
        changed, the indexing process is skipped unless the `force` parameter is set
        to True.

        Args:
            pdf_path (str): The absolute or relative path to the PDF file.
            force (bool, optional): If True, forces re-indexing even if the file has not changed. Defaults to False.
            batch_size (int, optional): The number of text chunks to process in a single batch. Defaults to 16.

        Raises:
            FileNotFoundError: If the PDF file specified in `pdf_path` does not exist.

        Returns:
            int: The total number of chunks indexed and stored in the vector store.
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

        print(f"Iniciando indexación para '{os.path.basename(pdf_path)}'...")
        self.vs.reset()

        texts_batch: List[str] = []
        metas_batch: List[Dict[str, Any]] = []
        total_chunks = 0

        # Auxiliary function for procecss and clean a batch.
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
            gc.collect()

        # Main loop that consumes from the generator.
        for text, meta in self._iter_pdf_chunks(pdf_path):
            texts_batch.append(text)
            metas_batch.append(meta)

            if len(texts_batch) >= batch_size:
                flush_batch()
        
        # Process the last batch.
        flush_batch()

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"sha256": current_hash, "chunks": total_chunks}, f)

        print(f"\n✅ Indexación completa. Total de {total_chunks} chunks guardados.")
        return total_chunks