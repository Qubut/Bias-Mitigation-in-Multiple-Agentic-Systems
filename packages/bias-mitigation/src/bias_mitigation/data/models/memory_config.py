"""Configuration schemas for memory intervention components (e.g. Mem0 vector store)."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """Configuration for LLM clients used by the memory system."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 8192
    top_p: float = 0.95
    openai_base_url: str | None = None
    api_key: SecretStr | None = None


class LLMProvider(BaseModel):
    """Defines an LLM provider and its associated configuration payload."""

    provider: str
    config: LLMConfig


class EmbedderConfig(BaseModel):
    """Configuration for text embedding models powering the retrieval store."""

    model: str
    ollama_base_url: str | None = None


class EmbedderProvider(BaseModel):
    """Defines an Embedding provider and its associated configuration payload."""

    provider: str
    config: EmbedderConfig


class VectorStoreConfig(BaseModel):
    """Configuration for the local or remote Vector Database."""

    path: str | None = None
    collection_name: str
    embedding_model_dims: int


class VectorStoreProvider(BaseModel):
    """Defines a Vector Store provider and its mapping."""

    provider: str
    config: VectorStoreConfig


class GraphStoreConfig(BaseModel):
    """Schema for Graph databases (e.g. Neo4j) to track entity-relations."""

    url: str
    username: str
    password: SecretStr


class GraphStoreProvider(BaseModel):
    """Graph store provider wrapper with concrete backend configuration."""

    provider: str
    config: GraphStoreConfig


class Mem0Config(BaseModel):
    """Validated top-level Mem0 runtime configuration payload."""

    llm: LLMProvider | None = Field(default=None, description='Language model configuration')
    embedder: EmbedderProvider | None = Field(
        default=None, description='Embedding model configuration'
    )
    vector_store: VectorStoreProvider | None = Field(
        default=None, description='Vector database backend'
    )
    graph_store: GraphStoreProvider | None = Field(
        default=None, description='Graph database backend'
    )
    reranker: dict[str, Any] | None = Field(
        default=None, description='Optional reranker (improves search relevance)'
    )

    class Config:
        """Pydantic behavior settings for Mem0 configuration model."""

        extra = 'allow'  # prevent unknown top-level keys
        validate_assignment = True
