import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import CLIPTextModel, CLIPTokenizer

load_dotenv()

token = os.getenv('MY_TOKEN')

print(token)

login(token)

# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the pipeline with optimizations
model_id = "stabilityai/stable-diffusion-3.5-large"
model_id = "stabilityai/stable-diffusion-3.5-large"

# Load components individually
unet = UNet2DConditionModel.from_pretrained(f"{model_id}/unet", use_safetensors=True, variant="fp16")
vae = AutoencoderKL.from_pretrained(f"{model_id}/vae", use_safetensors=True, variant="fp16")
text_encoder = CLIPTextModel.from_pretrained(f"{model_id}/text_encoder", use_safetensors=True, variant="fp16")
tokenizer = CLIPTokenizer.from_pretrained(f"{model_id}/tokenizer")
scheduler = PNDMScheduler.from_pretrained(f"{model_id}/scheduler")

# Initialize the pipeline with all components
pipe = StableDiffusionPipeline(
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

# Enable memory-efficient optimizations (if on CUDA)
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

# Enable attention slicing for lower VRAM usage
pipe.enable_attention_slicing()

# Set up a generator for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# Run the pipeline with optimized settings
pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape",
    negative_prompt="Oversaturated, blurry, low quality",
    height=1024, width=1024,
    guidance_scale=7.5,
    num_inference_steps=30,
    generator=generator
)

# View the resulting image
pipe_output.images[0].show()
