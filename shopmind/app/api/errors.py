class SearchApplicationError(Exception):
    """Base exception exposed above the application service boundary."""

class SearchInfrastructureError(SearchApplicationError):
    """Search storage/runtime dependency failed."""
class UnsupportedSearchMode(SearchApplicationError):
    """Requested retrieval strategy is not supported."""
class EmbeddingUnavailable(SearchApplicationError):
    """Embedding model/provider is not ready for a requested operation."""
class InvalidFilter(SearchApplicationError):
    """A requested public filter field is not supported."""
class InvalidImage(SearchApplicationError):
    """Uploaded bytes are not a supported decodable image."""
class ImageTooLarge(SearchApplicationError):
    """Uploaded image exceeds the configured request limit."""
class KnowledgeRetrievalUnavailable(SearchApplicationError):
    """RAG knowledge retrieval dependency failed."""
class RerankerUnavailable(SearchApplicationError):
    """Configured RAG reranker failed."""
class LLMUnavailable(SearchApplicationError):
    """Configured LLM provider failed."""
class RAGGenerationUpstreamError(SearchApplicationError):
    """LLM output remained invalid after bounded repair."""
