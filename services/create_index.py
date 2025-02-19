import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

from services.extract_pdf import extract_text_from_pdf

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index(pdf_path, index_save_path):
    # Extract text
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks (FAISS works best with small text blocks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)

    # Convert chunks into vector embeddings
    vectors = [embedding_model.embed_query(chunk) for chunk in text_chunks]

    # Create FAISS index
    d = len(vectors[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(vectors))

    # Store FAISS index
    faiss.write_index(index, index_save_path)
    print(f"FAISS index saved to {index_save_path}")


def search_faiss(query, index_path):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Convert query to embedding
    query_embedding = np.array([embedding_model.embed_query(query)])

    # Search FAISS for similar documents
    D, I = index.search(query_embedding, k=3)  # Top 3 matches
    print(f"Top matches: {I}, Distances: {D}")


# Example: Create FAISS index from a PDF
create_faiss_index("recipies.pdf", "faiss_index")
