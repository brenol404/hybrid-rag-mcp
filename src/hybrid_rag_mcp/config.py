from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    llm_model: str = "qwen3:8b"
    embed_batch_size: int = 32  # embeddings por request (medido: 16-32 é o ponto ótimo)

    cloud_base_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""

    corpus_dir: str = "examples/corpus"
    data_dir: str = "data"
    audit_log: str = "data/audit.jsonl"
    audit_max_bytes: int = 5_000_000  # rotaciona o audit.jsonl ao passar disso (0 = sem rotação)
    audit_keep: int = 3  # quantos backups (audit.jsonl.1, .2, ...) manter
    qdrant_path: str = "data/qdrant"
    qdrant_url: str = ""  # ex.: http://localhost:6333 — se setado, usa servidor (multi-processo)

    cache_enabled: bool = True
    cache_path: str = "data/cache.jsonl"
    cache_sim_threshold: float = 0.92
    cache_ttl_sec: int = 3600
    cache_max_entries: int = 256

    context_compression: int = 0

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    bm25_top_k: int = 10
    rrf_k: int = 60
    rrf_w_vector: float = 1.0
    rrf_w_lexical: float = 1.5
    rerank_model: str = ""
    rerank_budget: int = 20
    ask_max_iterations: int = 2

    mcp_auth_token: str = ""  # se setado, o transporte HTTP exige `Authorization: Bearer`


@lru_cache
def get_settings() -> Settings:
    return Settings()
