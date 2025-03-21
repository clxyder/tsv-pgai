from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgai.sqlalchemy import vectorizer_relationship

from config import Settings

class Base(DeclarativeBase):
    pass

class BlogPost(Base):
    __tablename__ = "blog_posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    content: Mapped[str]

    # Add vector embeddings for the content field
    content_embeddings = vectorizer_relationship(
        dimensions=Settings.embedding_dim,
    )
