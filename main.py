import asyncio
import time
import gc
import torch
import logging

from workflows.errors import WorkflowRuntimeError
import config


async def test_rag(queries: list[str]):
    from rag import AGENT, CONTEXT, QUERY_ENGINE

    rag_total_time = 0.0

    for query in queries:
        logging.info(f"Query:\n{query}")

        start = time.perf_counter()
        try:
            rag_response = await AGENT.run(
                query,
                ctx=CONTEXT,
                max_iterations=config.MAX_ITERATIONS,
            )
        except WorkflowRuntimeError as e:
            max_iterations_time = time.perf_counter()
            logging.warning(
                f"Max iterations reached after {max_iterations_time - start} seconds. Falling back to QueryEngine."
            )
            rag_response = await QUERY_ENGINE.aquery(query)
        end = time.perf_counter()

        elapsed = end - start
        rag_total_time += elapsed

        logging.info(f"RAG response:\n{str(rag_response).strip()}")
        logging.info(f"Time elapsed: {elapsed:.3f} seconds\n")

    logging.info(
        f"RAG agent time for {len(queries)} queries: {rag_total_time:.3f}s, avg: {rag_total_time/len(queries):.3f}s\n\n"
    )


def test_cag(queries: list[str]):
    from cag import cag_query

    torch.cuda.empty_cache()
    gc.collect()

    cag_total_time = 0.0

    for query in queries:
        logging.info(f"Query:\n{query}")

        start = time.perf_counter()
        cag_response = cag_query(query)
        end = time.perf_counter()

        elapsed = end - start
        cag_total_time += elapsed

        logging.info(f"CAG response:\n{cag_response.strip()}")
        logging.info(f"Time elapsed: {elapsed:.3f} seconds\n")

    logging.info(
        f"CAG time for {len(queries)} queries: {cag_total_time:.3f}s, avg: {cag_total_time/len(queries):.3f}s\n\n"
    )


def log_config():
    logging.info(
        f"""Configuration:
    MODEL_ID: {config.MODEL_ID}
    USE_4BIT_QUANTIZATION: {config.USE_4BIT_QUANTIZATION}
    MAX_NEW_TOKENS: {config.MAX_NEW_TOKENS}
    
    EMBED_MODEL_ID: {config.EMBED_MODEL_ID}
    MAX_ITERATIONS: {config.MAX_ITERATIONS}
    CHUNK_SIZE: {config.CHUNK_SIZE}
    CHUNK_OVERLAP: {config.CHUNK_OVERLAP}
    RETRIEVE_TOP_K: {config.RETRIEVE_TOP_K}
    TEMPERATURE: {config.TEMPERATURE}
    GENERATE_TOP_K: {config.GENERATE_TOP_K}
    GENERATE_TOP_P: {config.GENERATE_TOP_P}
    REPETITION_PENALITY: {config.REPETITION_PENALITY}
    
    RAG_SYSTEM_PROMPT: {config.RAG_SYSTEM_PROMPT}
    
    CAG_SYSTEM_PROMPT: {config.CAG_SYSTEM_PROMPT}
    """
    )


if __name__ == "__main__":
    from queries import queries

    logging.basicConfig(
        filename=f"./logs/{config.MODEL_ID.split('/')[1]}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not queries:
        logging.error("No queries found.")
        exit(1)

    log_config()

    logging.info("\n================\t Beginning CAG tests... \t================")
    test_cag(queries)

    logging.info("\n================\t Beginning RAG tests... \t================")
    asyncio.run(test_rag(queries))
