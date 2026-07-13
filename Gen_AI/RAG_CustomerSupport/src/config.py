import os
import streamlit as st
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

qdrantclient = QdrantClient(
    url=st.secrets["Qdrant_URL"],
    api_key=st.secrets["Qdrant_API"],
    cloud_inference=True
)

collection_name=st.secrets['collection_name']

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
    base_url=st.secrets['base_url'],
    api_key=st.secrets['OpenRouter']
)
