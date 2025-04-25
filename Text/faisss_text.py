import faiss
import numpy as np
import json
import pandas as pd

# Load text embeddings (assuming you've saved them previously)
text_embeddings = np.load(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/text_embeddings.npy")  # adjust path if needed

# Build FAISS index for text
index = faiss.IndexFlatL2(text_embeddings.shape[1])
index.add(text_embeddings)

# Save index
faiss.write_index(index, r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/text_index.faiss")

df = pd.read_csv(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/metadata.csv")  # must have 'image' and 'text' columns
metadata = df.to_dict(orient="records")
# Save metadata (optional but recommended)
with open(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/text_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
