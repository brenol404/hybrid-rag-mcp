"""Camada de otimização de tokens: cache semântico + compressão de contexto."""

from .cache import CacheHit, SemanticCache
from .compress import make_compressor, token_savings

__all__ = ["CacheHit", "SemanticCache", "make_compressor", "token_savings"]
