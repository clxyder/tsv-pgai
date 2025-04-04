
from dataclasses import dataclass


@dataclass
class Settings:

    model_name = 'nomic-embed-text'
    embedding_dim = 768

    db_url = "postgresql://postgres:postgres@localhost:5432/postgres"
    db_blog_table_name = "blog_posts"
    db_blog_chunk_column = "content"
    db_blog_target_table_name = "blog_embeddings"

    db_wiki_table_name = "wiki"
    db_wiki_chunk_column = "text"
    db_wiki_target_table_name = "wiki_embeddings"
