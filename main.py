import asyncio
import time

import config
import documents
import storage_context
from model import model, tokenizer
from queries import queries
from rag import agent, context
from cag import get_or_create_kv_cache, get_kv_len, run_cag


async def test_rag(queries: list[str]):
    rag_total_time = 0.0

    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        rag_response = await agent.run(query, ctx=context)
        end = time.perf_counter()

        elapsed = end - start
        rag_total_time += elapsed

        print(f"RAG:\n{rag_response}")
        print(f"Time taken: {elapsed:.3f} seconds")

    print(
        f"RAG total time: {rag_total_time:.3f}s, avg: {rag_total_time/len(queries):.3f}s"
    )


def test_cag(queries: list[str]):
    knowledge_cache = get_or_create_kv_cache(model, tokenizer)
    kv_len = get_kv_len(knowledge_cache)

    cag_total_time = 0.0

    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        cag_response = run_cag(model, tokenizer, knowledge_cache, kv_len, query)
        end = time.perf_counter()

        elapsed = end - start
        cag_total_time += elapsed

        print(f"CAG:\n{cag_response}")
        print(f"Time taken: {elapsed:.3f} seconds")

    print(
        f"CAG total time: {cag_total_time:.3f}s, avg: {cag_total_time/len(queries):.3f}s"
    )


if __name__ == "__main__":
    if not queries:
        print("No queries found.")
        exit(1)

    print("Starting RAG tests")
    asyncio.run(test_rag(queries))

    print("Starting CAG tests")
    test_cag(queries)
