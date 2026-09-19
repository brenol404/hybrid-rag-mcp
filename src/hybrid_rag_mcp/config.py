from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    llm_model: str = "qwen3:8b"

    cloud_base_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""

    corpus_dir: str = "examples/corpus"
    data_dir: str = "data"
    audit_log: str = "data/audit.jsonl"
    qdrant_path: str = "data/qdrant"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    bm25_top_k: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
