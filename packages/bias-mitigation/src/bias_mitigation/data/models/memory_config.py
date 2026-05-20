"""Configuration schemas for memory intervention components (e.g. Mem0 vector store)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


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
    openai_base_url: str | None = None
    api_key: SecretStr | None = None


class EmbedderProvider(BaseModel):
    """Defines an Embedding provider and its associated configuration payload."""

    provider: str
    config: EmbedderConfig


class VectorStoreConfig(BaseModel):
    """Configuration for the local or remote Vector Database."""

    path: str | None = None
    collection_name: str
    embedding_model_dims: int | None = None

    model_config = ConfigDict(extra='allow', validate_assignment=True)


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
    enable_dimension_fallback: bool = Field(
        default=True,
        description='If true, Mem0Tools may retry without explicit embedding dimensions.',
    )
    embedder_force_dimensionless_requests: bool = Field(
        default=True,
        description='If true, OpenAI-compatible embedder requests are sent without dimensions to avoid endpoint 400 mismatches.',
    )
    recall_top_k: int = Field(
        default=3,
        ge=1,
        description='Default number of memories to recall for each interaction step.',
    )
    max_query_chars: int = Field(
        default=2048,
        ge=128,
        description='Maximum characters allowed for memory search query text.',
    )
    max_memory_content_chars: int = Field(
        default=4096,
        ge=256,
        description='Maximum characters allowed for persisted memory content.',
    )
    max_passage_chars: int = Field(
        default=1200,
        ge=128,
        description='Maximum characters retained per recalled memory passage.',
    )
    max_recalled_chars: int = Field(
        default=4000,
        ge=256,
        description='Maximum total characters included in recalled memory block.',
    )
    enable_resilient_search_fallback: bool = Field(
        default=True,
        description='If true, recoverable retrieval failures degrade to empty memories.',
    )
    search_fallback_limit: int = Field(
        default=1,
        ge=1,
        description='Fallback search limit used during recoverable retrieval retries.',
    )
    search_fallback_retry_attempts: int = Field(
        default=1,
        ge=1,
        description='Max retry attempts for fallback search calls after a recoverable primary search failure.',
    )
    search_fallback_consecutive_fail_trip_threshold: int = Field(
        default=8,
        ge=1,
        description='Open fallback suppression circuit after this many consecutive fallback failures.',
    )
    search_fallback_cooldown_ms: int = Field(
        default=20000,
        ge=100,
        description='Fallback suppression cooldown duration in milliseconds after consecutive fallback failures trip.',
    )
    search_fallback_warning_every: int = Field(
        default=25,
        ge=1,
        description='Emit fallback warning log every N graceful-empty fallback events.',
    )
    memory_add_infer: bool = Field(
        default=False,
        description='If true, Mem0 uses LLM memory-action inference during add operations.',
    )
    memory_pressure_mode: str = Field(
        default='normal',
        description='Memory runtime mode: normal, read_only, or disabled.',
    )
    memory_operation_semaphore_limit: int = Field(
        default=1,
        ge=1,
        description='Max concurrent Mem0 operations per process.',
    )
    memory_slot_timeout_ms: int = Field(
        default=350,
        ge=10,
        description='Max wait in milliseconds for acquiring one memory operation slot.',
    )
    pressure_timeout_trip_threshold: int = Field(
        default=12,
        ge=1,
        description='Open backpressure circuit after this many slot timeouts.',
    )
    pressure_cooldown_ms: int = Field(
        default=12000,
        ge=100,
        description='Backpressure cooldown duration after timeout threshold is reached.',
    )
    drop_store_on_backpressure: bool = Field(
        default=True,
        description='If true, store operations are dropped gracefully while pressure circuit is open.',
    )
    degrade_search_on_backpressure: bool = Field(
        default=True,
        description='If true, search operations degrade to empty recall while pressure circuit is open.',
    )
    pressure_warning_every: int = Field(
        default=25,
        ge=1,
        description='Emit pressure-mode warning every N dropped/degraded operations.',
    )
    store_retry_attempts: int = Field(
        default=3,
        ge=1,
        description='Max attempts for transient store failures.',
    )
    search_retry_attempts: int = Field(
        default=3,
        ge=1,
        description='Max attempts for transient search failures.',
    )
    retry_backoff_min_ms: int = Field(
        default=50,
        ge=0,
        description='Initial retry backoff in milliseconds.',
    )
    retry_backoff_max_ms: int = Field(
        default=1500,
        ge=1,
        description='Maximum retry backoff in milliseconds.',
    )
    retry_jitter_ms: int = Field(
        default=120,
        ge=0,
        description='Random retry jitter in milliseconds.',
    )
    transient_error_markers: list[str] = Field(
        default_factory=lambda: [
            'cannot commit - no transaction is active',
            'database is locked',
            'transaction',
            'busy',
            'timeout',
            'temporarily unavailable',
            'deadline exceeded',
            'timed out',
            'connection reset',
            'connection refused',
            'service unavailable',
            'too many requests',
            '429',
            '502',
            '503',
            '504',
        ],
        description='Case-insensitive substrings used to classify transient backend failures.',
    )
    render_recalled_memory_style: str = Field(
        default='bullet',
        description='How recalled memory is rendered for prompts: bullet or plain.',
    )
    recall_max_items: int = Field(
        default=5,
        ge=1,
        description='Maximum recalled memory snippets rendered into the prompt.',
    )
    include_question_in_memory_text: bool = Field(
        default=True,
        description='If true, include the question text in stored memory content.',
    )

    model_config = ConfigDict(extra='allow', validate_assignment=True)

    def to_mem0_dict(self, strip_embedding_dims: bool = False) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True, mode='json')
        if self.llm and self.llm.config.api_key:
            data.setdefault('llm', {}).setdefault('config', {})['api_key'] = (
                self.llm.config.api_key.get_secret_value()
            )
        if self.embedder and self.embedder.config.api_key:
            data.setdefault('embedder', {}).setdefault('config', {})['api_key'] = (
                self.embedder.config.api_key.get_secret_value()
            )
        vector_store_payload = data.get('vector_store', {})
        vector_store_provider = str(vector_store_payload.get('provider', '')).lower()
        vector_store_config = vector_store_payload.get('config', {})
        should_strip_dims = strip_embedding_dims or vector_store_provider != 'qdrant'
        if should_strip_dims and isinstance(vector_store_config, dict):
            vector_store_config.pop('embedding_model_dims', None)
        return data
