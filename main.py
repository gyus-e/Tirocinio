import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import bitsandbytes as bnb
from dotenv import load_dotenv
import os
from cag import read_kv_cache, prepare_kvcache, write_kv_cache, clean_up, generate


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Configuration for 4-bit quantization using bitsandbytes.
# (Optional) can be passed to AutoModelForCausalLM.from_pretrained().
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)

kv_cache_path = "./kv_cache.pt"

if os.path.exists(kv_cache_path):
    knowledge_cache, kv_len = read_kv_cache(kv_cache_path)
else:
    from knowledge import knowledge

    knowledge_cache, kv_len = prepare_kvcache(model, tokenizer, documents=knowledge)
    write_kv_cache(knowledge_cache, kv_cache_path)


from querys import querys

for query in querys:
    clean_up(knowledge_cache, kv_len)
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = generate(model, input_ids, knowledge_cache)
    generated_text = tokenizer.decode(
        output[0], skip_special_tokens=True, temperature=None
    )
    print(f"Query\n: {query}")
    print(f"Response:\n {generated_text}")
