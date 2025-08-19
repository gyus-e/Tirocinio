import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
import bitsandbytes as bnb
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
embed_model_id = "BAAI/bge-m3"

# Configuration for 4-bit quantization using bitsandbytes.
# (Optional) can be passed to AutoModelForCausalLM.from_pretrained().
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
