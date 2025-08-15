import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from core.vectorstore import VectorStore


class ChromaStore(VectorStore):
    def __init__(self, persist_dir: str = ".chromadb", collection_name: str = "thesis"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings())
        self.collection = self.client.get_or_create_collection(name=collection_name)
        

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """_summary_

        Args:
            texts (List[str]): _description_
            metadatas (List[Dict[str, Any]]): _description_
            emmbeddings (List[List[float]]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[str]: _description_
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        return ids
    
    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        """_summary_

        Args:
            query_embedding (List[float]): _description_
            k (int, optional): _description_. Defaults to 4.

        Raises:
            NotImplementedError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        return {
            "documents": res.get("documents", [[]])[0],
            "metadatas": res.get("metadatas", [[]])[0],
            "distances": res.get("distances", [[]])[0],
        }