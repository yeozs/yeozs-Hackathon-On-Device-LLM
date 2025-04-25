from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Load and save processor
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(r"/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/model/smolv500")

# Load and save model
model = AutoModelForImageTextToText.from_pretrained(model_id)
model.save_pretrained(r"C:/Users/yeozuosheng/Documents/SIA_Hackathon/RAG/model/smolv500")
