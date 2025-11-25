import asyncio
import time
import gc
import torch
import logging

from workflows.errors import WorkflowRuntimeError
import config


async def test_rag(queries: list[str]):
    from rag import agent, context, query_engine

    rag_total_time = 0.0

    for query in queries:
        logging.info(f"Query:\n{query}\n")

        start = time.perf_counter()
        try:
            rag_response = await agent.run(
                query,
                ctx=context,
                max_iterations=config.max_iterations,
            )
        except WorkflowRuntimeError as e:
            logging.warning(e, "Falling back to QueryEngine...")
            rag_response = await query_engine.aquery(query)
        end = time.perf_counter()

        elapsed = end - start
        rag_total_time += elapsed

        logging.info(f"RAG:\n{rag_response}\n")
        logging.info(f"Time elapsed: {elapsed:.3f} seconds\n\n")

    logging.info(
        f"RAG agent time for {len(queries)} queries: {rag_total_time:.3f}s, avg: {rag_total_time/len(queries):.3f}s"
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


def log_config():
    logging.info("Configuration:")
    logging.info(f"model id: {config.model_id}")
    logging.info(f"max new tokens: {config.max_new_tokens}")
    logging.info(f"embed model id: {config.embed_model_id}")
    logging.info(f"max iterations: {config.max_iterations}")
    logging.info(f"chunk size: {config.chunk_size}")
    logging.info(f"chunk_overlap: {config.chunk_overlap}")
    logging.info(f"retrieve_top_k: {config.retrieve_top_k}")
    logging.info(f"temperature: {config.temperature}")
    logging.info(f"generate_top_k: {config.generate_top_k}")
    logging.info(f"generate_top_p: {config.generate_top_p}")
    logging.info(f"repetition_penalty: {config.repetition_penalty}")
    logging.info(f"rag system prompt: {config.rag_system_prompt}")
    logging.info(f"cag system prompt: {config.cag_system_prompt}")
    logging.info(f"cag answer instruction: {config.cag_answer_instruction}")


if __name__ == "__main__":
    from queries import queries

    logging.basicConfig(
        filename=f"./logs/{config.model_id.split('/')[1]}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log",
        level=logging.INFO,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
    )

    if not queries:
        logging.error("No queries found.")
        exit(1)

    log_config()

    logging.info("================\t Beginning CAG tests... \t================")
    test_cag(queries)

    logging.info("================\t Beginning RAG tests... \t================")
    asyncio.run(test_rag(queries))
