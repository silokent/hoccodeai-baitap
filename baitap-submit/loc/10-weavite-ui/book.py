# Viết code để tìm kiếm sách/query từ Weavite
import gradio as gr
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization

vector_db_client = weaviate.connect_to_local()

vector_db_client.connect()

print("DB is ready:", vector_db_client.is_ready())

COLLECTION_NAME = "BookCollection"


def search_book(query):
    book_collection = vector_db_client.collections.get(COLLECTION_NAME)
    response = book_collection.query.near_text(
        query=query, limit=10
    )

    results = []
    for book in response.objects:
            book_tuple = (
                book.properties.get('title', ''),
                book.properties.get('author', ''),
                book.properties.get('genre', ''),
                book.properties.get('description', ''))
            results.append(book_tuple)
    print(results)
    return results


with gr.Blocks(title="Tìm kiếm phim với Vector Database") as interface:
    query = gr.Textbox(label="Tìm kiếm sách", placeholder="Tên, tác giả, thể loại,...")
    search = gr.Button(value="Search")
    results = gr.Dataframe(
        headers=["Title", "Author", "Genre", "Description"],
        label="Kết quả tìm kiếm"
    )

    search.click(fn=search_book, inputs=query, outputs=results)

interface.queue().launch()