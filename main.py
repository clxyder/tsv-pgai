from time import sleep

from sqlalchemy import create_engine, func, text, exc
from sqlalchemy.orm import sessionmaker
from pgai.vectorizer import CreateVectorizer
from pgai.vectorizer.configuration import (
    EmbeddingLitellmConfig,
    EmbeddingOllamaConfig,
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
        source=Settings.db_table_name,
        target_table=Settings.db_target_table_name,
        embedding=EmbeddingOllamaConfig(
            model=Settings.model_name,
            dimensions=Settings.embedding_dim,
        ),
        chunking=ChunkingCharacterTextSplitterConfig(
            chunk_column=Settings.db_chunk_column,
            chunk_size=800,
            chunk_overlap=400,
            separator='.',
            is_separator_regex=False
        ),
        formatting=FormattingPythonTemplateConfig(template='$title - $chunk')
    ).to_sql()

    # execute vectorizer_statement
    r = session.execute(text(vectorizer_statement))
    session.commit()

    results = r.fetchall()
    for row in results:
        print(row)


def load_data():
        
    try:
        r = session.execute(text(
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

        print("Successfully loaded dataset.")
    except exc.IntegrityError:
        print("Dataset already loaded.")



def search(query: str):
    similar_posts = (
        session.query(BlogPost.content_embeddings)
        .order_by(
            BlogPost.content_embeddings.embedding.cosine_distance(
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
    # load_data()
    
    # create vectorizer
    create_vectorizer()
    
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
    