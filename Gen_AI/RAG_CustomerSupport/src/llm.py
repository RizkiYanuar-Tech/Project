from .config import llm_client, qdrantclient, collection_name
from .qdrantdb import embedding_model
from qdrant_client import models
import tensorflow as tf
import numpy as np
import openai
import pickle
import os

def load_model():
    src_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(src_path, '..', 'model', 'classification_model.keras')
    intent_vocab = os.path.join(src_path, '..', 'intent_vocab.pkl')

    if os.path.exists(model_path) and os.path.exists(intent_vocab):
        print("Model dan Vocabulary")
        model = tf.keras.models.load_model(model_path)
        with open(intent_vocab, 'rb') as f:
            intent_label = pickle.load(f)
    else:
        raise FileNotFoundError(
            "Model dan Vocabulary tidak ada, mungkin belum di training?"
        )
    return model, intent_label

def rag_pipeline(query_user, model, intent_label):
    # Receive input user to tensorflow model
    proba = model.predict(tf.constant([query_user]), verbose=0)
    highest_intent = np.argmax(proba[0])
    intent = intent_label[highest_intent]

    print(f"User ingin mengajukan pertanyaan mengenai {intent}")

    # Search in Qdrant Vector Database
    vector_query = embedding_model.encode(query_user).tolist()

    search_result = qdrantclient.query_points(
        collection_name=collection_name,
        query=vector_query,
        with_payload=True,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key='intent',
                    match=models.MatchValue(value=intent)
                ),
            ]
        ),
        limit=2
    ).points

    solusi = ""
    for idx, result in enumerate(search_result):
        solusi += f"solusi {idx + 1}. {result.payload.get('response')}\n"

    system_prompt=f"""
    You're a helpful, friendly Customer Support Assistant in E-Commerce
    Your job is handle customer complaint.

    RULES:
    1. ONLY Answer using information from [SOLUSI] block below
    2. ALWAYS answer based on data that store in Vector Database QDRANT, IF there no solution in [SOLUSI]
    dont make any answers from your hallucination, simply say: 'Mohon maaf, solusi untuk masalah ini tidak ditemukan'.
    3. ALWAYS answer in polite and natural indonesian language, regardless of the language used in the [SOLUSI] block
    4. IF there a multiple steps to complete the problem, present them clearly using a numbered list.
    
    [SOLUSI]
    {solusi}
    """
    try:
        completion = llm_client.chat.completions.create(
            model='poolside/laguna-xs-2.1:free',
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query_user
                }
            ],
            temperature=0.3,
            frequency_penalty=0.3
        )
        answers = completion.choices[0].message.content
        return answers
    except openai.RateLimitError as e:
        print(f"Gagal menggunakan {model} karena {e}")
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to {model} API: {e}")
        pass
    fail = "Sistem AI Sedang Sibuk, Coba beberapa saat lagi"
    return fail

if __name__=='__main__':
    query = input("Apa yang ingin ditanyakan?")
    model, intent_label = load_model()
    rag_pipeline(query, model, intent_label)
