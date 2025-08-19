import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import bitsandbytes as bnb
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
embed_model_id = "BAAI/bge-m3"
kv_cache_path = "./kv_cache.pt"

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
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


Settings.llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
Settings.chunk_size = 512
Settings.chunk_overlap = 64


async def main():

    from cag import read_kv_cache, prepare_kvcache, write_kv_cache, clean_up, generate
    from rag import agent

    if os.path.exists(kv_cache_path):
        knowledge_cache, kv_len = read_kv_cache(kv_cache_path)
    else:
        from knowledge import knowledge

        knowledge_cache, kv_len = prepare_kvcache(model, tokenizer, documents=knowledge)
        write_kv_cache(knowledge_cache, kv_cache_path)

    from querys import querys

    for query in querys:
        print(f"Query:\n{query}")

        rag_response = await agent.run(query)
        print(f"RAG:\n{rag_response}")

        clean_up(knowledge_cache, kv_len)
        input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
        output = generate(model, input_ids, knowledge_cache)
        cag_response = tokenizer.decode(
            output[0], skip_special_tokens=True, temperature=None
        )
        print(f"CAG:\n{cag_response}")


if __name__ == "__main__":
    asyncio.run(main())
