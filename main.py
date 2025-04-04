from typing import List
from time import sleep

from sqlalchemy import create_engine, func, text, exc, Sequence
from sqlalchemy.orm import sessionmaker

from pgai.vectorizer import CreateVectorizer
from pgai.vectorizer.configuration import (
    EmbeddingLitellmConfig,
    EmbeddingOllamaConfig,
    ChunkingCharacterTextSplitterConfig,
    FormattingPythonTemplateConfig,
)

from config import Settings
from models import Base, BlogPost, Wiki

# Create database engine and session
engine = create_engine(Settings.db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()


def create_vectorizer(table_name: str, target_table_name: str, chunk_column: str) -> None:

    vectorizer_created = False
    status = get_vectorizer_status()
    for row in status:
        if target_table_name in row[2]:
            vectorizer_created = True
            break

    if not vectorizer_created:
        vectorizer_statement = CreateVectorizer(
            source=table_name,
            target_table=target_table_name,
            embedding=EmbeddingOllamaConfig(
                model=Settings.model_name,
                dimensions=Settings.embedding_dim,
            ),
            chunking=ChunkingCharacterTextSplitterConfig(
                chunk_column=chunk_column,
                chunk_size=800,
                chunk_overlap=400,
                separator='.',
                is_separator_regex=False
            ),
            formatting=FormattingPythonTemplateConfig(template='$title - $chunk')
        ).to_sql()

        # execute vectorizer_statement
        session.execute(text(vectorizer_statement))
        session.commit()

        print(f"Successfully created vectorizer for {table_name}.")
    else:
        print(f"Vectorizer for {table_name} already created.")



def load_blog_data() -> None:

    entries = session.query(BlogPost).count()

    if entries > 0:
        print(f"Blog dataset already loaded with {entries} entries.")
    else:
        session.execute(text(
            """
            INSERT INTO blog_posts (id, url, title, authors, content, meta)
            VALUES
            (
                1,
                'https://www.postgresql.org/docs/current/tutorial-start.html',
                'Getting Started with PostgreSQL',
                'John Doe',
                'PostgreSQL is a powerful, open source object-relational database system...',
                '{"tags": ["database", "postgresql", "beginner"], "read_time": 5, "published_date": "2024-03-15"}'
            ),
            (
                2,
                'https://www.clearvoice.com/resources/improve-your-blog-writing/',
                '10 Tips for Effective Blogging',
                'Jane Smith, Mike Johnson',
                'Blogging can be a great way to share your thoughts and expertise...',
                '{"tags": ["blogging", "writing", "tips"], "read_time": 8, "published_date": "2024-03-20"}'
            ),
            (
                3,
                'https://aimagazine.com/machine-learning/alan-turing-a-strong-legacy-that-powers-modern-ai',
                'The Future of Artificial Intelligence',
                'Dr. Alan Turing',
                'As we look towards the future, artificial intelligence continues to evolve...',
                '{"tags": ["AI", "technology", "future"], "read_time": 12, "published_date": "2024-04-01"}'
            ),
            (
                4,
                'https://diabetes.org/food-nutrition/eating-healthy/tips-eating-healthy-on-go',
                'Healthy Eating Habits for Busy Professionals',
                'Samantha Lee',
                'Maintaining a healthy diet can be challenging for busy professionals...',
                '{"tags": ["health", "nutrition", "lifestyle"],
                "read_time": 6, "published_date": "2024-04-05"}'
            ),
            (
                5,
                'https://www.coursera.org/learn/introduction-to-cloud',
                'Introduction to Cloud Computing',
                'Chris Anderson',
                'Cloud computing has revolutionized the way businesses operate...',
                '{"tags": ["cloud", "technology", "business"], "read_time": 10, "published_date": "2024-04-10"}'
            ); 
            """)
        )
        session.commit()

        print("Successfully loaded blog dataset.")


def load_wiki_data() -> None:

    entries = session.query(Wiki).count()

    if entries > 0:
        print(f"Wikipedia dataset already loaded with {entries} entries.")
    else:
        session.execute(text(
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
        session.commit()

        print("Successfully loaded wiki dataset.")


def search_blogs(query: str) -> List:
    similar_posts = (
        session.query(BlogPost.content_embeddings)
        .order_by(
            BlogPost.content_embeddings.embedding.cosine_distance(
                func.ai.ollama_embed(
                    Settings.model_name,
                    query,
                )
            )
        )
        .limit(5)
        .all()
    )
    
    return similar_posts

def search_wiki(query: str) -> List:
    similar_posts = (
        session.query(Wiki.content_embeddings)
        .order_by(
            Wiki.content_embeddings.embedding.cosine_distance(
                func.ai.ollama_embed(
                    Settings.model_name,
                    query,
                )
            )
        )
        .limit(5)
        .all()
    )
    
    return similar_posts


def get_vectorizer_status() -> Sequence:
    # SELECT * from ai.vectorizer_errors
    r = session.execute(text("select * from ai.vectorizer_status;"))
    session.commit()

    return r.fetchall() 


def main():
    print("Hello from tsv-pgai!")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # load dataset
    load_blog_data()
    load_wiki_data()
    
    # create vectorizer
    create_vectorizer(
        Settings.db_blog_table_name,
        Settings.db_blog_target_table_name,
        Settings.db_blog_chunk_column,
    )

    create_vectorizer(
        Settings.db_wiki_table_name,
        Settings.db_wiki_target_table_name,
        Settings.db_wiki_chunk_column,
    )
    
    # run search
    ret = search_blogs("good food")

    print("blog search results:")
    for r in ret:
        # print(vars(r))
        # print(dir(r))
        print(f"* {r.chunk}")
        # print(f"{r.title} - {r.chunk}")

    
    ret = search_wiki("properties of light")

    print("wiki search results:")
    for r in ret:
        print(f"* {r.chunk}")
        # print(f"{r.title} - {r.chunk}")

if __name__ == "__main__":
    main()
