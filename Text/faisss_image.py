# faiss_index_images.py
import faiss
import numpy as np
import pandas as pd
import json

# Load your image embeddings
image_embeddings = np.load(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_embeddings.npy")  # shape (N, D)

# Create FAISS index
index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)

# Save FAISS index to disk
faiss.write_index(index, r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_index.faiss")

# Load metadata (assuming you already have your image-text CSV)
df = pd.read_csv(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/metadata.csv")  # must have 'image' and 'text' columns

# Convert metadata to list of dicts
metadata = df.to_dict(orient="records")

# Save metadata
with open(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ Image FAISS index and metadata saved.")
