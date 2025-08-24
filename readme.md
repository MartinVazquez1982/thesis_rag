# RAG Application with Qdrant & Streamlit

A complete Retrieval-Augmented Generation (RAG) application that allows you to chat with your documents. This project uses Qdrant as the vector database, a FastAPI backend for the core logic, and a Streamlit frontend for the user interface. The entire application is containerized with Docker for easy setup and deployment.

***

## 🏛️ Architecture

The application is composed of three main services orchestrated by **Docker Compose**:

* **`frontend`**: A [Streamlit](https://streamlit.io/) application that provides a user-friendly web interface. Users can upload documents and ask questions through this UI.
* **`backend`**: A [FastAPI](https://fastapi.tiangolo.com/) server that handles the core RAG logic. It processes documents, creates embeddings, stores them in Qdrant, and generates answers by querying the vector database and interacting with a language model.
* **`qdrant`**: A [Qdrant](https://qdrant.tech/) vector database instance. It is responsible for efficiently storing and searching through high-dimensional vector embeddings of the documents.

## 🛠️ Tech Stack

* **Backend**: Python, FastAPI, Uvicorn, Gemini
* **Frontend**: Python, Streamlit
* **Database**: Qdrant (Vector Database)
* **DevOps**: Docker, Docker Compose

***

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

You must have the following installed:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Clone the Repository

```bash
git clone https://github.com/MartinVazquez1982/thesis_rag
cd thesis_rag
```

### 2. Configure Environment Variables

The application uses .env files for configuration. You'll need to create one for the backend.

#### Backend Configuration:
Create a file named .env inside the backend/ directory (backend/.env). Copy the following content into it. This connects the backend to the Qdrant container.

```
# backend/.env
# === Proveedores (OBLIGATORIO) ===
LLM_PROVIDER=gemini
EMBEDDINGS_PROVIDER=gemini
VECTORSTORE_PROVIDER=qdrant

# === Gemini (según proveedor) ===
GEMINI_API_KEY=
GEMINI_EMBED_MODEL=models/embedding-001
GEMINI_LLM_MODEL=gemini-1.5-pro

# === Qdrant (según proveedor) ===
COLLECTION=thesis_rag
QDRANT_URL=http://qdrant:6333
# QDRANT_API_KEY=
```

#### Frontent Configuration:

```
# frontend/.env
RAG_URL=http://backend:8000
```

### 3. Build and Run the Application

From the project's root directory (where docker-compose.yml is located), run the following command:

```bash
docker-compose up --build -d
```

- --build: Builds the images for the first time or if you've made changes to the Dockerfile.
- -d: Runs the containers in detached mode (in the background).

The initial startup may take a few minutes as Docker downloads the necessary images and builds your application containers.

## 🖥️ Usage

Once the containers are running, you can access the different parts of the application:

- 🌐 Streamlit Frontend: Open your browser and go to http://localhost:8501

    - This is the main user interface for uploading files and asking questions.

- 🗂️ Qdrant Web UI: To inspect your vector collections, go to http://localhost:6333/dashboard

- 🔌 Backend API Docs: The FastAPI backend provides automatic API documentation. Access it at http://localhost:8000/docs

To stop the application, run the following command in the project root:

```
docker-compose down
```

## 🧩 Extensibility

Adding a new LLM, Embedder, or VectorStore
The architecture is designed to be extensible through interfaces and factories. To add a new provider:

1. Create a class that extends the corresponding interface:

    - LLM → e.g., MyCoolLLM(LLM)

    - Embedder → e.g., MyCoolEmbedder(Embedder)

    - VectorStore → e.g., MyCoolVectorStore(VectorStore)

2. Implement the methods required by the interface.

    - For example, generate(...) in LLM, embed(texts) in Embedder, or reset/query in VectorStore.

3. Register the provider in the appropriate factory:

    - Update the `get_llm`, `get_embeddings`, or `get_vectorstore` to map the new *_PROVIDER (the string from the .env file) to your class.

4. Configure environment variables:

    - Add any new required variables (tokens, endpoints, models) to the backend/.env file.

    - Adjust `LLM_PROVIDER`, `EMBEDDINGS_PROVIDER`, or `VECTORSTORE_PROVIDER` to point to your new provider (the name must match the one expected by the factory).