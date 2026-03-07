"""
Memory Store with Vector Embeddings

Provides long-term memory storage with semantic retrieval:
- ChromaDB for vector storage
- Sentence transformers for embeddings
- Semantic search for relevant context retrieval
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.context.message_types import AgentMessage


@dataclass
class Memory:
    """
    A single memory unit stored in the vector database.

    Memories are created from AgentMessages but optimized for retrieval.
    """

    id: str
    content: str
    embedding: Optional[List[float]] = None

    # Source information
    source_agent: str = "system"
    message_type: str = "thought"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Categorization
    category: str = "general"  # task, skill, observation, decision, etc.
    tags: List[str] = field(default_factory=list)

    # Importance and recency
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None

    # References
    related_skill: Optional[str] = None
    related_task: Optional[str] = None
    source_message_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: AgentMessage, category: str = "general") -> 'Memory':
        """Create a memory from an AgentMessage."""
        return cls(
            id=f"mem_{message.id}",
            content=message.to_embedding_text(),
            source_agent=message.source,
            message_type=message.message_type,
            timestamp=message.timestamp,
            category=category,
            importance=message.importance,
            source_message_id=message.id,
            metadata=message.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding)."""
        data = asdict(self)
        data.pop('embedding', None)  # Don't serialize embedding
        return data


class EmbeddingModel:
    """
    Wrapper for embedding model with fallback.

    Uses sentence-transformers if available, otherwise falls back
    to simple TF-IDF style embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(model_name)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self._model = None
                self.embedding_dim = 384  # Fallback dimension
        else:
            self.embedding_dim = 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Fallback: simple hash-based pseudo-embeddings
            # (Not semantic, but allows the system to function)
            embeddings = []
            for text in texts:
                # Create deterministic pseudo-embedding from text hash
                hash_bytes = hashlib.sha384(text.encode()).digest()
                embedding = [float(b) / 255.0 - 0.5 for b in hash_bytes]
                embeddings.append(embedding)
            return embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        return self.embed([text])[0]


class MemoryStore:
    """
    Vector-based memory store for long-term context.

    Features:
    - Semantic similarity search
    - Category-based filtering
    - Importance-weighted retrieval
    - Automatic memory consolidation
    """

    def __init__(
        self,
        persist_dir: Path = None,
        collection_name: str = "agent_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.collection_name = collection_name

        # Initialize embedding model
        self.embedder = EmbeddingModel(embedding_model)

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            if self.persist_dir:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self.client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False),
                )

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            # Fallback: in-memory storage
            self.client = None
            self.collection = None
            self._memories: Dict[str, Memory] = {}
            self._embeddings: Dict[str, List[float]] = {}

    def add(self, memory: Memory) -> str:
        """Add a memory to the store."""
        # Generate embedding if not present
        if memory.embedding is None:
            memory.embedding = self.embedder.embed_single(memory.content)

        if self.collection is not None:
            # ChromaDB storage
            self.collection.add(
                ids=[memory.id],
                embeddings=[memory.embedding],
                metadatas=[memory.to_dict()],
                documents=[memory.content],
            )
        else:
            # Fallback storage
            self._memories[memory.id] = memory
            self._embeddings[memory.id] = memory.embedding

        return memory.id

    def add_message(self, message: AgentMessage, category: str = "general") -> str:
        """Add an AgentMessage as a memory."""
        memory = Memory.from_message(message, category)
        return self.add(memory)

    def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        source_agent: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """
        Search for relevant memories.

        Returns list of (memory, similarity_score) tuples.
        """
        query_embedding = self.embedder.embed_single(query)

        if self.collection is not None:
            # Build where clause
            where = {}
            if category:
                where["category"] = category
            if source_agent:
                where["source_agent"] = source_agent

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Get extra for filtering
                where=where if where else None,
                include=["metadatas", "documents", "distances"],
            )

            memories = []
            for i, (meta, doc, dist) in enumerate(zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            )):
                # Filter by importance
                if meta.get("importance", 0.5) < min_importance:
                    continue

                memory = Memory(
                    id=results["ids"][0][i],
                    content=doc,
                    **{k: v for k, v in meta.items() if k in Memory.__dataclass_fields__}
                )

                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - dist

                memories.append((memory, similarity))

                if len(memories) >= n_results:
                    break

            return memories
        else:
            # Fallback: brute-force search
            results = []
            for mem_id, memory in self._memories.items():
                # Apply filters
                if category and memory.category != category:
                    continue
                if source_agent and memory.source_agent != source_agent:
                    continue
                if memory.importance < min_importance:
                    continue

                # Compute similarity
                mem_emb = self._embeddings.get(mem_id)
                if mem_emb:
                    similarity = self._cosine_similarity(query_embedding, mem_emb)
                    results.append((memory, similarity))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:n_results]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        if self.collection is not None:
            results = self.collection.get(
                ids=[memory_id],
                include=["metadatas", "documents"],
            )
            if results["ids"]:
                meta = results["metadatas"][0]
                return Memory(
                    id=memory_id,
                    content=results["documents"][0],
                    **{k: v for k, v in meta.items() if k in Memory.__dataclass_fields__}
                )
            return None
        else:
            return self._memories.get(memory_id)

    def update_importance(self, memory_id: str, importance: float):
        """Update the importance of a memory."""
        if self.collection is not None:
            self.collection.update(
                ids=[memory_id],
                metadatas=[{"importance": importance}],
            )
        elif memory_id in self._memories:
            self._memories[memory_id].importance = importance

    def delete(self, memory_id: str):
        """Delete a memory."""
        if self.collection is not None:
            self.collection.delete(ids=[memory_id])
        else:
            self._memories.pop(memory_id, None)
            self._embeddings.pop(memory_id, None)

    def get_by_category(self, category: str, limit: int = 100) -> List[Memory]:
        """Get all memories in a category."""
        if self.collection is not None:
            results = self.collection.get(
                where={"category": category},
                limit=limit,
                include=["metadatas", "documents"],
            )
            memories = []
            for i, (meta, doc) in enumerate(zip(
                results["metadatas"],
                results["documents"],
            )):
                memories.append(Memory(
                    id=results["ids"][i],
                    content=doc,
                    **{k: v for k, v in meta.items() if k in Memory.__dataclass_fields__}
                ))
            return memories
        else:
            return [m for m in self._memories.values() if m.category == category]

    def count(self) -> int:
        """Get total number of memories."""
        if self.collection is not None:
            return self.collection.count()
        return len(self._memories)

    def clear(self):
        """Clear all memories."""
        if self.collection is not None:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._memories.clear()
            self._embeddings.clear()
