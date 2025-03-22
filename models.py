from typing import Dict, Any

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import JSON
from pgai.sqlalchemy import vectorizer_relationship

from config import Settings

class Base(DeclarativeBase):
    type_annotation_map = {
        Dict[str, Any]: JSON
    }

class BlogPost(Base):
    __tablename__ = Settings.db_table_name

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str]
    title: Mapped[str]
    authors: Mapped[str]
    content: Mapped[str]
    meta: Mapped[Dict[str, Any]]

    # Add vector embeddings for the content field
    content_embeddings = vectorizer_relationship(
        dimensions=Settings.embedding_dim,
        target_table=Settings.db_target_table_name,
    )
