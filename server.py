import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv('MY_TOKEN')

print(token)

login(token)


# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the pipeline with optimizations
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Optimize memory usage
    safety_checker=None,  
).to(device)

# Enable memory-efficient optimizations (if on CUDA)
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

# Enable attention slicing for lower VRAM usage
pipe.enable_attention_slicing()

# Set up a generator for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# Run the pipeline with optimized settings
pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape",  # What to generate
    negative_prompt="Oversaturated, blurry, low quality",  # What NOT to generate
    height=512, width=640,  # Ensure dimensions are multiples of 64
    guidance_scale=7.5,  # Lower guidance scale for better creativity
    num_inference_steps=5,  # Reduce steps slightly for faster generation
    generator=generator  # Fixed random seed
)

# View the resulting image
pipe_output.images[0].show()