import os
import logging
import torch
import bitsandbytes as bnb
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from config import model_id

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    hf_token_path = os.getenv("HF_TOKEN_PATH")
    if hf_token_path and os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            hf_token = f.read().strip()

# Configuration for 4-bit quantization using bitsandbytes.
# (Optional) can be passed to AutoModelForCausalLM.from_pretrained().
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = "cuda" if torch.cuda.is_available() else "auto"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    device_map=device,
    token=hf_token,
)
logging.debug("Model loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
logging.debug("Tokenizer loaded successfully.")
