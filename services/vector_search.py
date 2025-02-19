from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Load pre-built FAISS index
faiss_index = FAISS.load_local("path_to_your_index", OpenAIEmbeddings())

def retrieve_context(query):
    docs = faiss_index.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])
