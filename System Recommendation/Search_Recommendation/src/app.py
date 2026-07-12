import xgboost as xgb
import pandas as pd
import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config import client

load_dotenv()

app = FastAPI(
    title='Wayfair Search',
    description='Search Product with Qdrant and Xgboost'
)

# get absolute folder location this file
src_direc = os.path.dirname(os.path.abspath(__file__))
# get root project location
project_root = os.path.dirname(src_direc)
model_dir = os.path.join(project_root, 'models')

print("Load Model Embedding & xgboost ranker...")
model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
rank = xgb.XGBRanker()
rank.load_model(os.path.join(model_dir, 'xgboost_ranker.json'))

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

def search_items(query_text: str, top_k: int):
    embeddings=model_embed.encode(query_text).tolist()
    results = client.query_points(
        collection_name='wayfair_items',
        query=embeddings,
        with_payload=True,
        limit=top_k
    )

    if not results.points:
        return pd.DataFrame()

    candidates = []
    for hit in results.points:
        payload=hit.payload
        payload['qdrant_score'] = hit.score
        candidates.append(payload)

    items = pd.DataFrame(candidates)

    items['query_length'] = len(query_text.split())
    items['query_class'] = np.nan

    numeric_cols = ['rating_count', 'review_count', 'average_rating']
    for col in numeric_cols:
        # Jika review_count tidak ada di Qdrant payload
        if col not in items.columns:
            items[col] = 0 
        items[col] = pd.to_numeric(items[col], errors='coerce').fillna(0)

    cat_cols = ['query_class', 'category_level_1', 'category_level_2', 'product_class']
    for cat in cat_cols:
        items[cat] = items[cat].astype('category')
    feature_data = [
        'rating_count',
        'review_count',
        'average_rating',
        'query_length',
        'query_class',
        'category_level_1',
        'category_level_2',
        'product_class'
    ]

    X_predict = items[feature_data]
    items['xgboost_score'] = rank.predict(X_predict)
    items_final = items.sort_values(by='xgboost_score', ascending=False).reset_index(drop=True)

    return items_final.head(top_k)

@app.get("/")
def home():
    return {"message": "/docs to use search function"}

@app.post("/search")
def search(request: SearchRequest):
    result = search_items(request.query, request.top_k)

    if result.empty:
        return {
            "query": request.query,
            "message": "Item not found in database",
            "result": []
        }

    cols_return = [
        "product_name", "average_rating", "product_description", "category_level_1", "category_level_2"
    ]
    coloms = [col for col in cols_return if col in result.columns]

    final_df = result[coloms].copy()

    for col in final_df.select_dtypes(['category']).columns:
        final_df[col] = final_df[col].astype(object)

    final_df = final_df.replace({np.nan: 'unknown'})
    
    # 4. Ubah ke Dictionary
    data = final_df.to_dict(orient="records")

    return {
        "query": request.query,
        "total_result": len(data),
        "result": data
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
