import gradio as gr
import os
import json
import numpy as np
from PIL import Image
import torch
import faiss
import open_clip
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import speech_recognition as sr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Config ===
FAISS_IMAGE_INDEX_PATH = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_index.faiss"
FAISS_TEXT_INDEX_PATH = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Text/text_index.faiss"
IMAGE_METADATA_JSON = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/image_metadata.json"
LLM_MODEL_PATH = r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/model/smolv500"
TOP_K = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Models Once ===
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()

processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
llm_model = AutoModelForImageTextToText.from_pretrained(
    LLM_MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# === Load FAISS Indexes and Metadata ===
image_index = faiss.read_index(FAISS_IMAGE_INDEX_PATH)
text_index = faiss.read_index(FAISS_TEXT_INDEX_PATH)
with open(IMAGE_METADATA_JSON, "r") as f:
    metadata = json.load(f)


def get_image_embedding(image):
    image = clip_preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    return embedding

# === Retrieve from FAISS ===
def retrieve_similar(index, query_vector, k=TOP_K, max_distance=1.0):
    query_vector_np = query_vector.cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_vector_np, k)
    filtered = [(i, d) for i, d in zip(indices[0], distances[0]) if d <= max_distance]
    if not filtered:
        return [], []
    indices_filtered, distances_filtered = zip(*filtered)
    return list(indices_filtered), list(distances_filtered)

# === LLM-based Answer Generator ===
def generate_response_with_image(image, context_text, question, matched=False):
    if matched:
        full_prompt = f"{question}\n\n{context_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt #(
                            #"You are a technical assistant specializing in aircraft maintenance.\n\n"
                            #"Below are historical maintenance logs related to an aircraft defect:\n\n"
                            #f"{context_text}\n\n"
                            #"Based on the above logs, please analyze the maintenance actions taken."
                        #)
                    }
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm_model.device, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)

    with torch.no_grad():
        generated_ids = llm_model.generate(**inputs, max_new_tokens=500)

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]

# === Main Query Pipeline ===
def query_pipeline_gradio(image, question):
    image_embedding = get_image_embedding(image)
    img_indices, _ = retrieve_similar(image_index, image_embedding)

    if not img_indices:
        # No relevant data
        fallback_question = "picture given is of aircraft part. please provide the details of the defect visible if any and maintenance required."
        return generate_response_with_image(image, "", fallback_question, matched=False)

    # Found related logs
    retrieved_texts = [metadata[i]['text'] for i in img_indices if i < len(metadata)]
    context_text = "\n".join(f"- {entry}" for entry in retrieved_texts)
    return generate_response_with_image(image, context_text, question, matched=True)


# === Speech-to-Text Function ===
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        # Using Google Web Speech API to transcribe
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

if __name__ == "__main__":
    with gr.Blocks(title="Aircraft Maintenance Assistant", css="""
        .header-img img { max-width: 100%; height: 20%; display: block; }
        #tiny-audio { height: 170px; }
        #tiny-image { height: 350px; }
        #scroll-output {
            overflow-y: scroll !important;
            max-height: 120px;}

    """) as demo:
        with gr.Column():
            # Header image
            gr.Image(value="/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/Images/SIAEC_Logo.png",
                     show_label=False,
                     show_download_button=False,
                     elem_classes=["header-img"])

            # Title and description
            gr.Markdown("## Aircraft Maintenance Assistant")
            gr.Markdown("Upload an image of a defective aircraft part and ask about maintenance history or suggestions.")

        #make image uploader and texbox side by side
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        elem_id="tiny-image"
                    )
                with gr.Column(scale=1):
                    #audio
                    audio_input = gr.Audio(
                        type="filepath",
                        label="",
                        show_label=False,
                        interactive=True,
                        elem_id="tiny-audio"
                    )
                    
                    question_input = gr.Textbox(
                        lines=2,
                        placeholder="Type your question or use the mic...",
                        label="Question"
                    )

                    # Automatically fill question box when audio is transcribed
                    audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=question_input)                    
                    submit_button = gr.Button("Submit")
                    
            with gr.Row():
                #with gr.Column(scale=1):
                output_text = gr.Textbox(label="Answer", lines=2, interactive=False, elem_id="scroll-output")

                # Click logic
                submit_button.click(
                    fn=query_pipeline_gradio,
                    inputs=[image_input, question_input],
                    outputs=output_text
                )


                

    #demo.launch(share=True)
    demo.launch()


