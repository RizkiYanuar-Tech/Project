import os
from config import client
from qdrant_client.models import Document
from dotenv import load_dotenv

load_dotenv()

def search_product(query_text, top_k=5):
    results = client.query_points(
        collection_name='wayfair_items',
        query=Document(text=query_text, model='sentence-transformers/all-MiniLM-L6-v2'),
        with_payload=True,
        limit=top_k
    )

    print("="*5)
    print(f"Top {top_k} hasil pencarian")
    print("="*5)

    if not results:
        print("Produk tidak ditemukan")
        return

    for result in results.points:
        print(f"Score: {result.score}")
        print(f"Product_id: {result.payload.get('product_id')}")
        print(f"Product_name: {result.payload.get('product_name')}")
        print(f"Product_description: {result.payload.get('product_description')}")
        print(f"Rating: {result.payload.get('average_rating')}")
        print("="*5)

if __name__ == "__main__":
    search_product("pillow arms")
