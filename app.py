from fastapi import FastAPI, File, UploadFile, Form
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from models.rag_model import generate_human_friendly_response

app = FastAPI()

# Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index"

TEXT_CHUNKS_PATH = "text_chunks.txt"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to create a FAISS index from text
def create_faiss_index(text_chunks):
    vectors = [embedding_model.embed_query(chunk) for chunk in text_chunks]
    
    d = len(vectors[0])  # Get the dimension of embeddings
    index = faiss.IndexFlatL2(d)  # Create FAISS index
    index.add(np.array(vectors))  # Add vectors to FAISS

    faiss.write_index(index, FAISS_INDEX_PATH)  # Save index to file

    # Save text chunks to a file
    with open(TEXT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n=====\n")

    return "FAISS index created successfully!"

# Function to load text chunks from the file
def load_text_chunks():
    if not os.path.exists(TEXT_CHUNKS_PATH):
        return []
    
    with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
        return f.read().split("\n=====\n")  # Splitting based on the delimiter

# API Endpoint to upload a PDF and index it
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(await file.read())  # Extract text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)  # Split text into chunks

    create_faiss_index(text_chunks)  # Store in FAISS

    return {"message": f"PDF {file.filename} indexed successfully!"}

# API Endpoint to search for a query in FAISS
@app.get("/search/")
async def search_faiss(query: str):
    if not os.path.exists(FAISS_INDEX_PATH):
        return {"error": "No FAISS index found. Upload a document first!"}

    index = faiss.read_index(FAISS_INDEX_PATH)  # Load FAISS index
    text_chunks = load_text_chunks()
    query_embedding = np.array([embedding_model.embed_query(query)])  # Convert query to embedding

    D, I = index.search(query_embedding, k=3)  # Search FAISS (Top 3 matches)
    # Get matched text using retrieved indices
    retrieved_texts = [text_chunks[i] for i in I[0] if i < len(text_chunks)]
        # Generate human-readable response
    response = generate_human_friendly_response(retrieved_texts)
    return {"response": response}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
