import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ingestion

# -----------------------------
# Load documents and chunks
# -----------------------------

DATA_PATH = "data/raw"

documents = ingestion.load_documents(DATA_PATH)
chunks = ingestion.chunk_documents(documents)

texts = [chunk["text"] for chunk in chunks]

# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load FAISS index
# -----------------------------

index = faiss.read_index("vector_index.faiss")

# -----------------------------
# Retriever function
# -----------------------------

def retrieve(query, k=3):

    # convert query to embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    # search FAISS index
    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(texts[i])

    return results


# -----------------------------
# Test the retriever
# -----------------------------

query = "Are we compliant with encryption policy?"

results = retrieve(query)

print("\nTop Retrieved Chunks:\n")

for r in results:
    print(r)
    print("\n----------------------\n")
