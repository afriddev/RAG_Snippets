from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

app = FastAPI()

# Load model once — this is critical for speed!
# model = SentenceTransformer('all-MiniLM-L6-v2')

model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')


# Predefined texts + embeddings
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Embeddings are numerical representations of text.",
    "Large language models are used in various AI tasks.",
    "Quantization helps models run faster on low-end devices."
]

# Precompute embeddings once
text_embeddings = [model.encode(text).reshape(1, -1) for text in texts]

@app.get("/similarity")
def get_similarity(query: str):
    start = time.time()
    query_vec = model.encode(query).reshape(1, -1)

    # Compute cosine similarity with each precomputed vector
    similarities = [float(cosine_similarity(query_vec, vec)[0][0]) for vec in text_embeddings]

    elapsed = (time.time() - start) * 1000  # ms
    return {
        "query": query,
        "similarities": list(zip(texts, similarities)),
        "time_ms": round(elapsed, 2)
    }
