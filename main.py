from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from pgai.vectorizer import CreateVectorizer
from pgai.vectorizer.configuration import (
    EmbeddingLitellmConfig,
    ChunkingCharacterTextSplitterConfig,
    FormattingPythonTemplateConfig,
)

from config import Settings
from models import BlogPost, Base

# Create database engine and session
engine = create_engine(Settings.db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Create tables
Base.metadata.create_all(bind=engine)

vectorizer_statement = CreateVectorizer(
    source="blog",
    target_table='blog_embeddings',
    embedding=EmbeddingLitellmConfig(
        # model='openai/text-embedding-3-small',
        model='ollama/nomic-embed-text',
        dimensions=Settings.embedding_dim,
    ),
    chunking=ChunkingCharacterTextSplitterConfig(
        chunk_column='content',
        chunk_size=800,
        chunk_overlap=400,
        separator='.',
        is_separator_regex=False
    ),
    formatting=FormattingPythonTemplateConfig(template='$title - $chunk')
).to_sql()

# TODO execute vectorizer_statement

similar_posts = (
    session.query(BlogPost.content_embeddings)
    .order_by(
        BlogPost.content_embeddings.embedding.cosine_distance(
            func.ai.openai_embed(
                "text-embedding-3-small",
                "search query",
                text("dimensions => 768")
            )
        )
    )
    .limit(5)
    .all()
)

def main():
    print("Hello from tsv-pgai!")


if __name__ == "__main__":
    main()
