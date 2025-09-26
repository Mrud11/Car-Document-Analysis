import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import json
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_texts_from_folder(folder_path, save_file="extracted_texts.pkl"):
    if os.path.exists(save_file):
        print(f"Loading saved texts from {save_file}")
        with open(save_file, "rb") as f:
            return pickle.load(f)
    
    texts = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for i, fname in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Extracting text from {fname}...")
        fpath = os.path.join(folder_path, fname)
        texts.append(extract_text_from_pdf(fpath))
    
    # Save for future runs
    with open(save_file, "wb") as f:
        pickle.dump(texts, f)
    
    return texts

# ---------------- Text Chunking ----------------
def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


# ---------------- ChromaDB Setup ----------------
client = chromadb.Client(Settings())
collection_name = "manual_chunks"
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(collection_name)
else:
    collection = client.create_collection(collection_name)

# ---------------- Embedding ----------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # lighter model

def embed_and_add_to_db(chunks, batch_size=8, save_file="chunk_embeddings.pkl"):
    if os.path.exists(save_file):
        print(f"Loading saved embeddings from {save_file}")
        with open(save_file, "rb") as f:
            embeddings = pickle.load(f)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"chunk_{i}"}],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )
        return embeddings

    embeddings = []
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start+batch_size]
        print(f"Embedding batch {start}-{start+len(batch)} of {total}")
        try:
            batch_embeddings = embed_model.encode(batch).tolist()
        except Exception as e:
            print(f"Error during embedding batch {start}-{start+len(batch)}: {e}")
            break
        embeddings.extend(batch_embeddings)
        for i, (chunk, embedding) in enumerate(zip(batch, batch_embeddings), start=start):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"chunk_{i}"}],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )
    with open(save_file, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

if __name__ == "__main__":
    pdf_folder = r"D:\\Users\\Mirdula\\Documents\\P-Projects\\Car\\Accent_1.6L_2005.pdf"  # <-- Must be a folder, not a single PDF file

    texts = extract_texts_from_folder(pdf_folder)
    full_text = "\n".join(texts)

    chunks = chunk_text(full_text, max_chars=1000)
    print(f"Total chunks: {len(chunks)}")

    embed_and_add_to_db(chunks)

    print("Welcome to Hyundai Accent Car Manual Q&A chatbot! Type 'exit' to quit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = answer_question(user_question)
        print(f"Answer: {answer}")
