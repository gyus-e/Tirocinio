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
from config import MODEL_ID, USE_4BIT_QUANTIZATION

load_dotenv()
__HF_TOKEN = os.getenv("HF_TOKEN")
if not __HF_TOKEN:
    __HF_TOKEN_PATH = os.getenv("HF_TOKEN_PATH")
    if __HF_TOKEN_PATH and os.path.exists(__HF_TOKEN_PATH):
        with open(__HF_TOKEN_PATH, "r") as f:
            __HF_TOKEN = f.read().strip()

# Configuration for 4-bit quantization using bitsandbytes.
# (Optional) can be passed to AutoModelForCausalLM.from_pretrained().
__BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

DEVICE = "cuda" if torch.cuda.is_available() else "auto"

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=__BNB_CONFIG if USE_4BIT_QUANTIZATION else None,
    device_map=DEVICE,
    token=__HF_TOKEN,
)
logging.debug("Model loaded successfully.")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
logging.debug("Tokenizer loaded successfully.")

EMBED_DEVICE = (
    "cuda" if torch.cuda.is_available() else MODEL.model.embed_tokens.weight.device
)
