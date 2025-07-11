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
