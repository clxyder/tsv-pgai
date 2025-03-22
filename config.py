
from dataclasses import dataclass


@dataclass
class Settings:

    model_name = 'nomic-embed-text'
    embedding_dim = 768

    db_url = "postgresql://postgres:postgres@localhost:5432/postgres"
    db_table_name = "blog_posts"
    db_chunk_column = "content"
    db_target_table_name = "blog_embeddings"
