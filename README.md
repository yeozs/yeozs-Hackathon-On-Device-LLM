# LLM Efficiency and Cost at Scale — A Case for On-Device Models

In his 1865 work "The Coal Question", English economist William Stanley Jevons observed that the increased efficiency of coal-powered steam engines did not correspond with a reduction in coal consumption. On the contrary, as technological advancements made the per unit use of coal more cost-effective, the overall demand for coal actually rose, resulting in greater total consumption and cost. This counterintuitive outcome, where improved efficiency leads to increased resource use and cost, is now known as Jevons Paradox.

This oddity is especially relevant in the age of Generative AI. While the per token cost for Large Language Models (LLMs) has decreased significantly over the years, overall usage — and therefore total cost — continues to grow.

At our company, this trend is evident. As adoption increases, so do total API token costs and infrastructure demands, particularly when relying on private LLM servers to meet internal security requirements.

We believe fine-tuned, on-device LLMs present a powerful opportunity to:

- Reduce reliance on costly private servers

- Lower API token consumption

- Maintain compliance with enterprise security standards

By running efficient, open-source models locally, we aim to make LLM usage cost-effective, secure, and scalable.

Disclaimer:
All files and code in this repository utilise open-source tools, models, and images.

<img width="951" alt="Screenshot 2025-04-25 at 10 47 22 AM" src="https://github.com/user-attachments/assets/eb38930f-4e1e-4465-a296-17b984fd447f" />

<p align="center"> Figure 1: Gradio Application running on Mac using offline SmolVLM-500M model </p>

----------------------------------------------
Please follow the below steps to utilise Gradio Code:

Step 1: Download all files in this Github Repository into folder called RAG

Step 2: create a virtual environment `python -m venv venv`

Step 3: Activate virtual environment `source venv/bin/activate`

Step 4: Install python requirements `pip install -r requirements.txt`

Step 5: Go to following HuggingFace URL and download model files. Save in folder called "model": https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct

Step 6: Go to Text file `cd Text`

Step 7: Replace all the filepaths in gradio_ui.py with your current RAG file path. To check RAG file path, go to RAG folder and use `pwd`

Step 8: Run `python gradio_ui.py`. Sample images is found in "Image" folder

Step 9: Deactivate virtual environment when done `Deactivate`

----------------------------------------------
Code Explanation:

Code utilises RAG - Retrieval Augmented Generation. 

1) "Images" file contains a database of Aircraft Maintenance defect images stored in image embedding form.

2) When a new image is uploaded in Gradio, the new image is matched to closest hit in image database.

3) The maintenance description text housed together with image file paths will  be retrieved. Together with user input text in Gradio, both texts are passed as input to LLM (SmolVLM-500M).

4) LLM Response is displayed in Gradio App.
----------------------------------------------
Contributors:
Dr Shoeb Shaikh, Mr Rahul Sharma, Mr Yeo Zuosheng, Mr Erik Cundomanik
