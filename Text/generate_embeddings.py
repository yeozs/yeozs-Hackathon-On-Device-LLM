
import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
import open_clip

# === Paths ===
CSV_PATH = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/metadata.csv"
IMAGE_DIR = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images"

# === Load CLIP ===
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in df["image"]]
input(image_paths)

# === Embed images ===
image_embeddings = []

for path in image_paths:
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_image(image)
        image_embeddings.append(embedding[0].numpy())
    except Exception as e:
        print(f"❌ Error with {path}: {e}")
        image_embeddings.append(np.zeros(512))  # default fallback

# === Save embeddings ===
np.save(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_embeddings.npy", np.vstack(image_embeddings))
print("✅ Image embeddings saved to image_embeddings.npy")
