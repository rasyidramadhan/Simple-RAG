# Simple-RAG
Retrieval-Augmented Generation (RAG) API, used to store documents, perform semantic-based searches, and generate contextual answers to user queries.

This RAG API is a simple implementation of a Retrieval-Augmented Generation system based on:
- Flask as the REST API,
- Qdrant as the vector database,
- SentenceTransformer for text embedding,
- OpenAI GPT as the generative model.

Architecture:
User → Flask API → SentenceTransformer → Qdrant → OpenAI → Response

Example qdrant running (Docker method):
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
