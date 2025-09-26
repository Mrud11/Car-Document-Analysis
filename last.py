import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_texts_from_folder(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.pdf'):
            fpath = os.path.join(folder_path, fname)
            print(f"Extracting text from {fname}")
            texts.append(extract_text_from_pdf(fpath))
    return texts

pdf_folder = r"D:\\Users\\Mirdula\\Documents\\P-Projects\\Car\\Documents"  # <-- Your document folder path here
texts = extract_texts_from_folder(pdf_folder)
full_text = "\n".join(texts)

def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(full_text, max_chars=1000)
print(f"Total chunks: {len(chunks)}")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.Client(Settings())
collection = client.create_collection("manual_chunks")

embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Changed embedding model here

def embed_texts(texts):
    return embed_model.encode(texts).tolist()

embeddings = embed_texts(chunks)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[chunk],
        metadatas=[{"source": f"chunk_{i}"}],
        embeddings=[embedding]
    )
print("Chunks added to ChromaDB")

# ---- Replace llama with OpenAI chat completions using Groq API ----
import requests

API_KEY = "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"   # <-- Insert your Groq API key here
API_URL = "https://api.groq.ai/v1/chat/completions"

def get_top_chunks(question, top_k=5):
    q_embed = embed_model.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )
    return [doc for doc in results['documents'][0]]

def answer_question(question):
    context = "\n".join(get_top_chunks(question))
    messages = [
        {"role": "system", "content": "You are an expert assistant for Hyundai Accent car manual. Use the context to answer the question clearly."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gpt-4o-mini",  # Example free Groq GPT model
        "messages": messages,
        "max_tokens": 256
    }
    response = requests.post(API_URL, json=data, headers=headers)
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Example usage
question = "How do I change a Hyundai Accent tire?"
print("Answer:", answer_question(question))

if __name__ == "__main__":
    print("Welcome to Hyundai Accent Car Manual Q&A chatbot!")
    print("Type 'exit' to quit.")

    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        answer = answer_question(user_question)
        print(f"Answer: {answer}")
