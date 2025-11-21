import asyncio
import time
import gc
import torch
import logging
import config


async def test_rag(queries: list[str]):
    from rag import query_engine, agent, context

    rag_total_time = 0.0

    for query in queries:
        logging.info(f"Query:\n{query}\n")

        start = time.perf_counter()
        # rag_response = await agent.run(query, ctx=context)
        rag_response = await query_engine.aquery(query)
        end = time.perf_counter()

        elapsed = end - start
        rag_total_time += elapsed

        logging.info(f"RAG:\n{rag_response}\n")
        logging.info(f"Time elapsed: {elapsed:.3f} seconds\n\n")

    logging.info(
        f"RAG time for {len(queries)} queries: {rag_total_time:.3f}s, avg: {rag_total_time/len(queries):.3f}s"
    )


def test_cag(queries: list[str]):
    from cag import get_or_create_kv_cache, get_kv_len, run_cag, clean_up

    torch.cuda.empty_cache()
    gc.collect()

    knowledge_cache = get_or_create_kv_cache(config.kv_cache_path)
    kv_len = get_kv_len(knowledge_cache)

    cag_total_time = 0.0

    for query in queries:
        logging.info(f"Query:\n{query}\n")

        start = time.perf_counter()
        cag_response = run_cag(knowledge_cache, query)
        end = time.perf_counter()

        elapsed = end - start
        cag_total_time += elapsed

        clean_up(knowledge_cache, kv_len)
        logging.info(f"CAG:\n{cag_response}\n")
        logging.info(f"Time elapsed: {elapsed:.3f} seconds\n\n")

    logging.info(
        f"CAG time for {len(queries)} queries: {cag_total_time:.3f}s, avg: {cag_total_time/len(queries):.3f}s"
    )


if __name__ == "__main__":
    from queries import queries

    logging.basicConfig(
        filename=f"./logs/{time.strftime("%Y%m%d-%H%M%S")}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not queries:
        logging.error("No queries found.")
        exit(1)

    logging.info("================\t Beginning RAG tests... \t================")
    asyncio.run(test_rag(queries))

    logging.info("================\t Beginning CAG tests... \t================")
    test_cag(queries)
