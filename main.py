import os
import replicate
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the REPLICATE_API_TOKEN from environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Ensure the token is set
if REPLICATE_API_TOKEN is None:
    raise ValueError("REPLICATE_API_TOKEN is not set. Please set it in the .env file.")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def generate_image(prompt, width, height, num_inference_steps):
    input = {
        "width": width,
        "height": height,
        "prompt": prompt,
        "refine": "expert_ensemble_refiner",
        "apply_watermark": False,
        "num_inference_steps": num_inference_steps
    }

    output = replicate.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input=input
    )
    return output[0]  # return the image URL

# Streamlit interface
st.title("Image Generator using Stable Diffusion and LangChain")

prompt = st.text_input("Prompt", "Enter a prompt for the image...")
width = st.slider("Width", min_value=256, max_value=1024, step=1, value=768)
height = st.slider("Height", min_value=256, max_value=1024, step=1, value=768)
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=50, step=1, value=25)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image_url = generate_image(prompt, width, height, num_inference_steps)
            st.image(image_url, caption="Generated Image")
    else:
        st.warning("Please enter a prompt to generate an image.")
