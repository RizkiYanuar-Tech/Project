from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Qdrant Cloud
client = QdrantClient(
    url='https://fe10d59c-045d-4bdd-aff6-b0a74d3d82e5.australia-southeast1-0.gcp.cloud.qdrant.io',
    api_key=os.getenv('API_KEY_QDRANT'),
    cloud_inference=True,
    timeout=60.0
)

# Create Collection
if not client.collection_exists(collection_name='wayfair_items'):
    client.create_collection(
        collection_name='wayfair_items',
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("Collection berhasil dibuat")
else:
    print("Collection sudah tersedia!")
