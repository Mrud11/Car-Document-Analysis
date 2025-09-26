#extract PDfs

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

pdf_folder = (r"D:\\Users\\Mirdula\\Documents\\P-Projects\\Car\\Accent_1.6L_2005.pdf")  # folder where you placed your downloaded PDFs
texts = extract_texts_from_folder(pdf_folder)
full_text = "\n".join(texts)

# chunk the Text for Embedding
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

#Embed and Store Chunks in ChromaDB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB (local, default settings)
client = chromadb.Client(Settings())
collection = client.create_collection("manual_chunks")

# Embedding model - fast and lightweight
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# 5. Q&A Retrieval and Answering Using TinyLlama
from llama_cpp import Llama

# Initialize TinyLlama (update with your actual model path)
llm = Llama(model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048, n_threads=8)

def get_top_chunks(question, top_k=5):
    q_embed = embed_model.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )
    return [doc for doc in results['documents'][0]]

def answer_question(question):
    context = "\n".join(get_top_chunks(question))
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm(prompt, max_tokens=256)
    return result['choices'][0]['text']

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
