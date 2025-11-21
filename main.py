import asyncio
import time
import gc
import torch
from config import kv_cache_path


async def test_rag(queries: list[str]):
    from rag import query_engine, agent, context

    rag_total_time = 0.0

    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        # rag_response = await agent.run(query, ctx=context)
        rag_response = await query_engine.aquery(query)
        end = time.perf_counter()

        elapsed = end - start
        rag_total_time += elapsed

        print(f"RAG:\n{rag_response}")
        print(f"Time elapsed: {elapsed:.3f} seconds")

    print(
        f"RAG time for {len(queries)} queries: {rag_total_time:.3f}s, avg: {rag_total_time/len(queries):.3f}s"
    )


def test_cag(queries: list[str]):
    from cag import get_or_create_kv_cache, get_kv_len, run_cag, clean_up

    torch.cuda.empty_cache()
    gc.collect()

    knowledge_cache = get_or_create_kv_cache(kv_cache_path)
    kv_len = get_kv_len(knowledge_cache)

    cag_total_time = 0.0

    for query in queries:
        print(f"Query:\n{query}")

        start = time.perf_counter()
        cag_response = run_cag(knowledge_cache, query)
        end = time.perf_counter()

        elapsed = end - start
        cag_total_time += elapsed

        clean_up(knowledge_cache, kv_len)
        print(f"CAG:\n{cag_response}")
        print(f"Time elapsed: {elapsed:.3f} seconds")

    print(
        f"CAG time for {len(queries)} queries: {cag_total_time:.3f}s, avg: {cag_total_time/len(queries):.3f}s"
    )


if __name__ == "__main__":
    from queries import queries

    if not queries:
        print("No queries found.")
        exit(1)

    print("Starting RAG tests")
    asyncio.run(test_rag(queries))

    print("Starting CAG tests")
    test_cag(queries)
