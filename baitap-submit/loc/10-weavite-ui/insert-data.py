import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization


COLLECTION_NAME = "BookCollection"

vector_db_client = weaviate.connect_to_local()

vector_db_client.connect()

print("DB is ready:", vector_db_client.is_ready())


def create_collection():

    book_collection = vector_db_client.collections.create(

        name=COLLECTION_NAME,

        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),

        properties=[
            # Tiêu đề phim: text, được vector hóa và chuyển thành chữ thường
        Property(
            name="title",
            data_type=DataType.TEXT,
            tokenization=Tokenization.LOWERCASE
        ),
        Property(
            name="author",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD
        ),
        Property(
            name="description",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD
        ),
        Property(
            name="intro",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD
        ),
        Property(
            name="excerpt",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD
        ),
        Property(
            name="notes",
            data_type=DataType.TEXT
        ),
        Property(
            name="genre",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD
        ),
        Property(
            name="grade",
            data_type=DataType.TEXT,
            skip_vectorization=True
        ),
        Property(
            name="lexile",
            data_type=DataType.TEXT,
            skip_vectorization=True
        ),
        Property(
            name="is_prose",
            data_type=DataType.BOOL,
            skip_vectorization=True
        ),
        Property(
            name="date",
            data_type=DataType.TEXT,
        ),
        Property(
            name="path",
            data_type=DataType.TEXT,
            skip_vectorization=True
        ),
        Property(
            name="license",
            data_type=DataType.TEXT,
            skip_vectorization=True
        ),
        ]
    )

    data = pd.read_csv("commonlit_texts.csv")

    data = data.fillna("")

    records = data.to_dict(orient="records")

    print(f"Inserting {len(records)} records...")

    with book_collection.batch.dynamic() as batch:

        for row in records:

            print(f"Inserting: {row['title']}")

            # Fix bool conversion
            is_prose_value = False

            if str(row["is_prose"]) in ["1", "true", "True"]:
                is_prose_value = True

            batch.add_object(
                properties={
                    "title": str(row["title"]),
                    "author": str(row["author"]),
                    "description": str(row["description"]),
                    "intro": str(row["intro"]),
                    "excerpt": str(row["excerpt"]),
                    "notes": str(row["notes"]),
                    "genre": str(row["genre"]),
                    "grade": str(row["grade"]),
                    "lexile": str(row["lexile"]),
                    "is_prose": is_prose_value,
                    "date": str(row["date"]),
                    "path": str(row["path"]),
                    "license": str(row["license"]),
                }
            )

    print("Data saved to Vector DB")

if vector_db_client.collections.exists(COLLECTION_NAME):
    print(f"{COLLECTION_NAME} already exists")

else:
    create_collection()

vector_db_client.close()