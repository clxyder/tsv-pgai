from time import sleep

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from pgai.vectorizer import CreateVectorizer
from pgai.vectorizer.configuration import (
    EmbeddingLitellmConfig,
    ChunkingCharacterTextSplitterConfig,
    FormattingPythonTemplateConfig,
)

from config import Settings
from models import Wiki, Base

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
    r = session.execute(text(vectorizer_statement))
    results = r.fetchall()
    for row in results:
        print(row)


def load_wiki_data():
    r = session.execute(text(
        """
        SELECT ai.load_dataset(
            'wikimedia/wikipedia',
            '20231101.en',
            table_name=>'wiki',
            batch_size=>5,
            max_batches=>1,
            if_table_exists=>'append'
        );
        """)
    )
    results = r.fetchall()
    for row in results:
        print(row)


def search(query: str):
    similar_posts = (
        session.query(Wiki.content_embeddings)
        .order_by(
            Wiki.content_embeddings.embedding.cosine_distance(
                func.ai.embedding_litellm(
                    Settings.model_name,
                    query,
                    text(f"dimensions => {Settings.embedding_dim}")
                )
            )
        )
        .limit(5)
        .all()
    )
    
    print(similar_posts)


def check_vectorizer_status():
    # SELECT * from ai.vectorizer_errors
    r = session.execute(text("select * from ai.vectorizer_status;"))
    results = r.fetchall()
    for row in results:
        print(row)


def main():
    print("Hello from tsv-pgai!")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # load dataset
    load_wiki_data()
    
    # create vectorizer
    # create_vectorizer()
    
    # try:
    #     while True:
    #         check_vectorizer_status()
    #         sleep(5)
    # except KeyboardInterrupt:
    #     pass
    
    # run search
    # search("properties of light")

if __name__ == "__main__":
    main()
    