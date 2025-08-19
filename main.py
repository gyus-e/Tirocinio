import asyncio
from model import model, tokenizer
from querys import querys
from rag import agent, context
from cag import get_or_create_kv_cache, run_cag

kv_cache_path = "./kv_cache.pt"


async def main():

    for query in querys:
        print(f"Query:\n{query}")
        rag_response = await agent.run(query, ctx=context)
        print(f"RAG:\n{rag_response}")

    knowledge_cache, kv_len = get_or_create_kv_cache(model, tokenizer, kv_cache_path)
    for query in querys:
        print(f"Query:\n{query}")
        cag_response = run_cag(model, tokenizer, knowledge_cache, kv_len, query)
        print(f"CAG:\n{cag_response}")


if __name__ == "__main__":
    asyncio.run(main())
