"""
Vector Store - ChromaDB integration for semantic search.
Provides embedding generation and similarity search for RAG.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.memory.vector_store")


class VectorStore:
    """
    ChromaDB-based vector store for semantic memory search.

    Features:
    - Automatic embedding generation
    - Similarity search with scores
    - Persistent storage
    - Async-safe operations
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.settings = get_settings()
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._lock = asyncio.Lock()

        # Persistence directory
        if persist_dir:
            self._persist_dir = Path(persist_dir)
        else:
            base = Path(self.settings.get("data.base_dir", "~/.openclaw")).expanduser()
            self._persist_dir = base / "chromadb"

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def initialize(self):
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                # Initialize persistent client
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )

                # Get or create collection
                collection_name = self.settings.get("memory.vector.collection", "openclaw_memory")
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                # Set up embedding function
                self._setup_embedding_function()

                self._initialized = True
                logger.info(f"VectorStore initialized with {self._collection.count()} documents")

            except ImportError:
                logger.error("chromadb not installed. Run: pip install chromadb")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize VectorStore: {e}")
                raise

    def _setup_embedding_function(self):
        """Configure the embedding function based on available providers."""
        embedding_provider = self.settings.get("memory.vector.embedding_provider", "default")

        if embedding_provider == "openai":
            try:
                import openai
                api_key = self.settings.get("providers.openai.api_key", "")
                self._embedding_fn = OpenAIEmbedder(api_key)
                logger.info("Using OpenAI embeddings")
                return
            except ImportError:
                pass

        if embedding_provider == "sentence-transformers" or embedding_provider == "default":
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.settings.get(
                    "memory.vector.model",
                    "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
                )
                self._embedding_fn = SentenceTransformerEmbedder(model_name)
                logger.info(f"Using SentenceTransformers embeddings: {model_name}")
                return
            except ImportError:
                pass

        # Fallback to simple hash-based embeddings (not semantic, but works)
        logger.warning("No embedding provider available, using fallback")
        self._embedding_fn = FallbackEmbedder()

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def add(self, content: str, metadata: dict = None, doc_id: str = None) -> str:
        """Add a document to the vector store."""
        await self.initialize()

        if not doc_id:
            doc_id = self._generate_id(content)

        embedding = await self._get_embedding(content)

        async with self._lock:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}]
            )

        logger.debug(f"Added document {doc_id} to vector store")
        return doc_id

    async def add_batch(self, documents: list[dict]) -> list[str]:
        """Add multiple documents at once."""
        await self.initialize()

        ids = []
        embeddings = []
        contents = []
        metadatas = []

        for doc in documents:
            content = doc.get("content", "")
            doc_id = doc.get("id") or self._generate_id(content)
            ids.append(doc_id)
            contents.append(content)
            metadatas.append(doc.get("metadata", {}))
            embeddings.append(await self._get_embedding(content))

        async with self._lock:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

        logger.info(f"Added {len(ids)} documents to vector store")
        return ids

    async def search(self, query: str, top_k: int = 5, where: dict = None) -> list[dict]:
        """Search for similar documents."""
        await self.initialize()

        query_embedding = await self._get_embedding(query)

        async with self._lock:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert cosine distance to similarity score (0-1)
                similarity = 1 - distance

                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(similarity, 4),
                })

        return formatted

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        await self.initialize()

        async with self._lock:
            try:
                self._collection.delete(ids=[doc_id])
                return True
            except Exception as e:
                logger.error(f"Failed to delete {doc_id}: {e}")
                return False

    async def clear(self):
        """Clear all documents from the collection."""
        await self.initialize()

        async with self._lock:
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"}
            )

        logger.info("Vector store cleared")

    async def count(self) -> int:
        """Get the number of documents in the store."""
        await self.initialize()
        return self._collection.count()

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if asyncio.iscoroutinefunction(self._embedding_fn.embed):
            return await self._embedding_fn.embed(text)
        return self._embedding_fn.embed(text)


# ── Embedding Providers ──────────────────────────────────────

class SentenceTransformerEmbedder:
    """Embedding using sentence-transformers (local, no API)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class OpenAIEmbedder:
    """Embedding using OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


class FallbackEmbedder:
    """Simple fallback embedder using TF-IDF-like hashing (not truly semantic)."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        import hashlib
        # Create pseudo-embedding by hashing tokens
        tokens = text.lower().split()
        embedding = [0.0] * self.dimensions

        for token in tokens:
            token_hash = hashlib.md5(token.encode()).digest()
            for i, byte in enumerate(token_hash):
                idx = (i * 16 + byte) % self.dimensions
                embedding[idx] += 0.1

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
