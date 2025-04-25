import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModelForVision2Seq, AutoProcessor ,AutoModelForImageTextToText
import open_clip

# === Config ===
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FAISS_IMAGE_INDEX_PATH = r"C:\Users\Rahul_Sharma\Downloads\Images\image_index.faiss"
FAISS_TEXT_INDEX_PATH = r"C:\Users\Rahul_Sharma\Downloads\Text\text_index.faiss"
IMAGE_METADATA_JSON = r"C:\Users\Rahul_Sharma\Downloads\Images\image_metadata.json"
LLM_MODEL_PATH = r"C:\Users\Rahul_Sharma\Downloads\model\smolvm"
TOP_K = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
#clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
#clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()


processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
llm_model = AutoModelForImageTextToText.from_pretrained(
    LLM_MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# === Load FAISS indexes and metadata ===
image_index = faiss.read_index(FAISS_IMAGE_INDEX_PATH)
text_index = faiss.read_index(FAISS_TEXT_INDEX_PATH)
with open(IMAGE_METADATA_JSON, "r") as f:
    metadata = json.load(f)
    print(metadata)

def get_image_embedding(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    #embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding 


'''def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True)'''

# === Retrieve from FAISS ===
def retrieve_similar(index, query_vector, k=TOP_K, max_distance=50.0):
    query_vector_np = query_vector.cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_vector_np, k)
    print("📊 All image distances:", distances[0])
    # Filter out results beyond the max distance
    filtered = [(i, d) for i, d in zip(indices[0], distances[0]) if d <= max_distance]
    
    if not filtered:
        return [], []

    indices_filtered, distances_filtered = zip(*filtered)
    return list(indices_filtered), list(distances_filtered)

# === LLM-based Answer Generator ===
def generate_response_with_image(image_path, context_text,question,matched=False):
    #image = Image.open(image_path).convert("RGB")
    input(context_text)
    if matched:
       messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a technical assistant specializing in aircraft maintenance.\n\n"
                    "Below are historical maintenance logs related to an aircraft defect:\n\n"
                    f"{context_text}\n\n"
                    "Based on the above logs, please analyze the maintenance actions taken in past and also provide the short explanation"
                )
            }
        ]
    }
]

    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": f"{question}"}
                ]
            }
        ]
   
    inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",

).to(llm_model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        generated_ids = llm_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.8, top_p=0.9, num_return_sequences=1)

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]

# === Main Query Pipeline ===
def query_pipeline(image_path, question):
    text_indices = None
    img_indices = None
    if image_path!="":
        print("🔍 Embedding query image...")
        image_embedding = get_image_embedding(image_path)
        print("🔍 Retrieving similar entries image...")
        img_indices, _ = retrieve_similar(image_index, image_embedding)
    else:
        print("🔍 Embedding query text...")
        text_embedding = get_text_embedding(question)
        print("🔍 Retrieving similar entries text...")
        text_indices, _ = retrieve_similar(text_index, text_embedding)

    all_indices = set()
    if img_indices is not None:
        all_indices = set(img_indices)
    if text_indices is not None:
        all_indices = set(text_indices)
    

    if not all_indices:
        print("⚠️ No relevant data found.")
        #question="picture given is of aircraft part. please provide the details of the defect vsisble if any and maintenance required."
        context_text=""
        final_response = generate_response_with_image(image_path, context_text, question,False)
        print("\n✅ Final Response:\n", final_response)
        return

    print("📄 Fetching relevant logs...")
    retrieved_texts = []

    for i in all_indices:
        if i < len(metadata):
            log = metadata[i]
            retrieved_texts.append(f"{log['text']}")

    context_text = "\n".join(f"- {entry}" for entry in retrieved_texts)
    print("🔍 Context text:\n", context_text)

    print("🧠 Generating final response with image + logs...")
    final_response = generate_response_with_image(image_path, context_text, question,True)

    print("\n✅ Final Response:\n", final_response)

# === Example usage ===
if __name__ == "__main__":
    query_image_path = r"C:\Users\Rahul_Sharma\Downloads\Images\wing_crack_01.png"  # Path to your query image
    user_question = "dis you notice any defect, if yes then please provide the details of the defect vsisble if any and maintenance required."
    query_pipeline(query_image_path, user_question)
