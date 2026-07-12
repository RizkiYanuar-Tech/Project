from dotenv import load_dotenv
import pandas as pd
from .config import qdrantclient, collection_name
import os
from qdrant_client.models import PointStruct
from qdrant_client.models import PayloadSchemaType
from sentence_transformers import SentenceTransformer

load_dotenv()

embedding_model=SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def upload_to_qdrant(data, batch_process=100):
    points=[]
    uploaded_count = 0
    total_rows=len(data)

    for idx, row in data.iterrows():
        # Vector Embedding Instruction Columns
        vector = embedding_model.encode(row['instruction'].tolist())

        point = PointStruct(
            id=idx,
            vector=vector,
            payload={
                "instruction": row['instruction'],
                "category": str(row['category']),
                "intent": str(row['intent']),
                "response": row['response']
            }
        )
        points.append(point)

        # Upsert points to collection
        # if point contain > 100 data
        if len(points) > batch_process:
            # Upload to Qdrant
            qdrantclient.upsert(
                collection_name=collection_name,
                points=points
            )
            points=[]
            uploaded_count += len(points)

        if len(points) < batch_process:
            qdrantclient.upsert(
                collection_name=collection_name,
                points=points
            )
            uploaded_count += len(points)

    # Indexing searching filter based on intent user
    qdrantclient.create_payload_index(
        collection_name=collection_name,
        field_name='intent', # Columns for indexing
        field_schema=PayloadSchemaType.KEYWORD # data type keyword
    )

    return {
        "status": "success",
        "total_data": total_rows,
        "total_data_upload": uploaded_count
    }

if __name__ == "__main__":
    src_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(src_path, '..', 'data')
    take_data = os.path.join(data_path, "dataset_clean.parquet")
    data = pd.read_parquet(take_data)
    upload_to_qdrant(data)
