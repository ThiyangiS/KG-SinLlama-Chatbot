"""Paths and defaults for the existing Sigiriya corpus and model stack."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
E5_MODEL = "intfloat/multilingual-e5-base"
SINLLAMA_MODEL = "SAWithanage/SinLlama-Llama-3-8B-Merged"
TEXT_INDEX = "sigiriya_chunk_vectors"
HEADING_INDEX = "sigiriya_heading_vectors"
VECTOR_LIMIT = 22
RRF_K = 60
TOP_K = 3


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str = field(repr=False)
    database: str = "neo4j"

    @classmethod
    def from_env(cls):
        """Read Neo4j settings at connection time."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.environ["NEO4J_PASSWORD"],
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
