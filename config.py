
from dataclasses import dataclass


@dataclass
class Settings:

    embedding_dim = 768
    db_url = "postgresql://postgres:postgres@localhost:5432/postgres"
    model_name = 'ollama/nomic-embed-text'
