import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from core.vectorstore import VectorStore # Asumo que esta es tu clase base

class QdrantStore(VectorStore):
    def __init__(self, collection_name: str = "thesis", vector_size: int = 768):
        self.collection_name = collection_name
        self.vector_size = vector_size # ¡Importante! Qdrant necesita saber el tamaño de tus embeddings

        # --- CAMBIO CLAVE ---
        # Inicializa el cliente para que guarde los datos en una carpeta local.
        # Esto es el equivalente a PersistentClient de ChromaDB, pero más optimizado.
        self.client = QdrantClient(url="http://localhost:6333")
        
        # Obtenemos la lista de colecciones existentes
        existing_collections = [c.name for c in self.client.get_collections().collections]

        # Creamos la colección SÓLO si no existe
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, # Tamaño del embedding (Gemini suele usar 768)
                    distance=models.Distance.COSINE,
                    on_disk=True # <-- ¡MAGIA! Fuerza que los vectores se guarden en disco.
                )
            )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Añade un lote de textos, metadatos y embeddings a la colección."""
        ids = [str(uuid.uuid4()) for _ in texts]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=[{**meta, "text": txt} for meta, txt in zip(metadatas, texts)]
            ),
            wait=False # Hacemos la subida asíncrona para no bloquear el script
        )
        return ids
    
    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        """Realiza una búsqueda de similitud."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True, # Para que nos devuelva los metadatos y el texto
            with_vectors=False # No necesitamos los vectores en la respuesta
        )
        
        # Formateamos la salida para que sea igual a la que tenías
        documents, metadatas, distances = [], [], []
        for hit in search_result:
            # El texto lo guardamos en el payload
            payload = hit.payload or {}
            documents.append(payload.pop("text", ""))
            metadatas.append(payload)
            distances.append(hit.score)
            
        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
    
    def reset(self) -> None:
        """Borra y recrea la colección."""
        self.client.delete_collection(collection_name=self.collection_name)
        # La colección se recreará automáticamente en el __init__ si es necesario,
        # pero es mejor ser explícito.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                on_disk=True
            )
        )