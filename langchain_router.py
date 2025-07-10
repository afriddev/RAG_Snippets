from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

doc_sources = [
    {
        "name": "leave",
        "summary": "Leave policy, number of casual leaves, carry forward rules, leave approval process.",
        "pdf_path": "./Leave_Policy.pdf",
        "table": "leave_records"
    },
    {
        "name": "privacy",
        "summary": "Employee privacy policy, data protection, third-party sharing, personal information security.",
        "pdf_path": "./Privacy_Policy.pdf",
        "table": "privacy_logs"
    },
    {
        "name": "conduct",
        "summary": "Code of conduct, employee ethics, rules on misconduct, punctuality, harassment.",
        "pdf_path": "./Code_of_Conduct.pdf",
        "table": "discipline_cases"
    },
]

# Step 2: Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
summary_texts = [doc["summary"] for doc in doc_sources]
summary_embeddings = embedding_model.embed_documents(summary_texts)

prompt_template = """
You are a helpful assistant answering questions about company policies.
Use only the context below to answer the question.
Give a short and accurate response. Do not guess.

Context:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

llm = Ollama(model="gemma3:1b")  # or tinyllama

questions = [
    "What are the employee policy"
]


for q in questions:
    q_embedding = embedding_model.embed_query(q)
    scores = cosine_similarity([q_embedding], summary_embeddings)[0]
    
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_doc = doc_sources[best_idx]

    if best_score > 0.5:
        loader = PyPDFLoader(best_doc["pdf_path"])
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(chunks, embedding=embedding_model)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt}
        )
        answer = qa.run(q)
        print("Answer:", answer)
    else:
        print(f"[Info] Not relevant in PDFs. You can query the database table: {best_doc['table']}")
