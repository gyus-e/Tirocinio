import asyncio
import time
from model import model, tokenizer
from queries import queries


async def test_rag(queries: list[str]):
    from rag import agent, context

    rag_total = 0.0
    
    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        rag_response = await agent.run(query, ctx=context)
        end = time.perf_counter()

        elapsed = end - start
        rag_total += elapsed

        print(f"RAG:\n{rag_response}")
        print(f"Time taken: {elapsed:.3f} seconds")

    print(f"RAG total: {rag_total:.3f}s, avg: {rag_total/len(queries):.3f}s")


def test_cag(queries: list[str]):
    from cag import get_or_create_kv_cache, get_kv_len, run_cag
    knowledge_cache = get_or_create_kv_cache(model, tokenizer)
    kv_len = get_kv_len(knowledge_cache)

    cag_total = 0.0

    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        cag_response = run_cag(model, tokenizer, knowledge_cache, kv_len, query)
        end = time.perf_counter()

        elapsed = end - start
        cag_total += elapsed

        print(f"CAG:\n{cag_response}")
        print(f"Time taken: {elapsed:.3f} seconds")

    print(f"CAG total: {cag_total:.3f}s, avg: {cag_total/len(queries):.3f}s")


if __name__ == "__main__":
    if not queries:
        print("No queries found.")
        exit(1)

    print("Starting RAG tests...")
    asyncio.run(test_rag(queries))

    print("\nStarting CAG tests...")
    test_cag(queries)
