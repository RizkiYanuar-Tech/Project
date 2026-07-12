import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

qdrantclient = QdrantClient(
    url=os.getenv("Qdrant_URL"),
    api_key=os.getenv("Qdrant_API"),
    cloud_inference=True
)

collection_name=os.getenv('collection_name')

if qdrantclient.collection_exists(collection_name=collection_name):
    print("Collection already exists")
else:
    qdrantclient.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )

llm_client=OpenAI(
    base_url=os.getenv("base_url"),
    api_key=os.getenv("OpenRouter")
)