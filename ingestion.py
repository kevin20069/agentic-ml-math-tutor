import os
import re
from typing import List, Dict
import os

DATA_PATH = os.path.join("data", "raw")
def load_documents(data_path: str) -> List[Dict]:

    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            full_path = os.path.join(data_path, file)

            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "filename": file,
                "text": text
            })

    print(f"Loaded {len(documents)} documents.")
    return documents

def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - remove extra spaces
    - normalize newlines
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def chunk_documents(documents: List[Dict], 
                    chunk_size: int = 500, 
                    overlap: int = 50) -> List[Dict]:
    """
    Chunks each document and attaches metadata
    """
    all_chunks = []

    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })

    print(f"Created {len(all_chunks)} chunks.")
    return all_chunks


if __name__ == "__main__":
    DATA_PATH = "data/raw"

    docs = load_documents(DATA_PATH)
    chunks = chunk_documents(docs)

    print("\nExample Chunk:\n")
    print(chunks[0])

