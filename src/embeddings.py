import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# import chunking pipeline
DATA_PATH = "data/raw"
import ingestionn

documents = ingestionn.load_documents(DATA_PATH)
chunks = ingestionn.chunk_documents(documents) 

# -----------------------------
# 1️⃣ Load and Chunk Documents
# -----------------------------


texts = [chunk["text"] for chunk in chunks]

print("Total chunks:", len(texts))


# -----------------------------
# 2️⃣ Load Embedding Model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 3️⃣ Generate Embeddings
# -----------------------------

print("Generating embeddings...")

embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# -----------------------------
# 4️⃣ Build FAISS Index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Total vectors in index:", index.ntotal)


# -----------------------------
# 5️⃣ Save Index
# -----------------------------

faiss.write_index(index, "vector_index.faiss")


print("Vector index saved!") 
