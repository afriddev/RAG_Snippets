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















""""



from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

q1 = "What is the capital of India?"
q2 = "Where is New Delhi located?"

q1_embedding = embedding_model.embed_query(q1)
q2_embedding = embedding_model.embed_query(q2)

vec1 = np.array(q1_embedding).reshape(1, -1)
vec2 = np.array(q2_embedding).reshape(1, -1)

similarity = cosine_similarity(vec1, vec2)[0][0]

print(f"Cosine Similarity: {similarity:.4f}")


"""
