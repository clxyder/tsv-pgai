from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgai.sqlalchemy import vectorizer_relationship

from config import Settings

class Base(DeclarativeBase):
    pass

class Wiki(Base):
    __tablename__ = "wiki"

    id: Mapped[str] = mapped_column(primary_key=True)
    url: Mapped[str]
    title: Mapped[str]
    text: Mapped[str]

    # Add vector embeddings for the content field
    content_embeddings = vectorizer_relationship(
        dimensions=Settings.embedding_dim,
    )
