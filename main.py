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



def create_vectorizer():
    vectorizer_statement = CreateVectorizer(
        source="wiki",
        target_table='wiki_embeddings',
        embedding=EmbeddingLitellmConfig(
            model=Settings.model_name,
            dimensions=Settings.embedding_dim,
        ),
        chunking=ChunkingCharacterTextSplitterConfig(
            chunk_column='text',
            chunk_size=800,
            chunk_overlap=400,
            separator='.',
            is_separator_regex=False
        ),
        formatting=FormattingPythonTemplateConfig(template='$title - $chunk')
    ).to_sql()

    # execute vectorizer_statement
    r = session.execute(vectorizer_statement)
    print(r)


def load_wiki_data():
    r = session.execute(
        """
        SELECT ai.load_dataset(
            'wikimedia/wikipedia',
            '20231101.en',
            table_name=>'wiki',
            batch_size=>5,
            max_batches=>1,
            if_table_exists=>'append'
        );
        """
    )
    print(r)

def search():
    similar_posts = (
        session.query(BlogPost.content_embeddings)
        .order_by(
            BlogPost.content_embeddings.embedding.cosine_distance(
                func.ai.openai_embed(
                    Settings.model_name,
                    "search query",
                    text("dimensions => 768")
                )
            )
        )
        .limit(5)
        .all()
    )
    
    print(similar_posts)


def main():
    print("Hello from tsv-pgai!")
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # create vectorizer
    create_vectorizer()
    
    # load dataset
    
    
    # run search


if __name__ == "__main__":
    main()
    