from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from config import client
import os

def upload_to_qdrant(product_data):
    model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Make sure no NaN value
    product_data['search_document'] = product_data['search_document'].fillna("")

    # Batching data to qdrant
    batch_size = 500
    total_product = len(product_data)

    print(f"Embedding data and upload {total_product} product")

    for i in tqdm(range(0, total_product, batch_size), desc='Upload to Qdrant'):
        # batch data
        batch_df = product_data.iloc[i:i + batch_size]

        # text to vector
        docs = batch_df['search_document'].tolist()
        embeddings=model_embed.encode(docs)

        # PointStruck
        points=[]
        for j, (_, row) in enumerate(batch_df.iterrows()):
            point_id=int(row['product_id'])

            # Metadata for return Qdrant
            payload={
                "product_id":point_id,
                "product_name": str(row['product_name']),
                "product_class": str(row['product_class']),
                "category_level_1": str(row['category_level_1']),
                "category_level_2": str(row['category_level_2']),
                "product_description": str(row['product_description']),
                "rating_count": str(row['rating_count']),
                "average_rating": str(row['average_rating']),
            }

            point=PointStruct(
                id=point_id,
                vector=embeddings[j].tolist(),
                payload=payload
            )
            points.append(point)

        client.upsert(
            collection_name='wayfair_items',
            points=points
        )

    print("Data success upload to Qdrant Cloud")

if __name__ == '__main__':
   # get absolute folder location this file
    src_direc = os.path.dirname(os.path.abspath(__file__))

    # get root project location
    project_root = os.path.dirname(src_direc)

    data_direc = os.path.join(project_root, 'data', 'clean')
    product_clean_path = os.path.join(data_direc, 'product_clean.parquet')
    product_clean = pd.read_parquet(product_clean_path)
    upload_to_qdrant(product_clean)
