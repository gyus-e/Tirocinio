import asyncio
import torch
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache

from src.cag.init_cag_settings import init_cag_settings
from src.rag.init_rag_settings import init_rag_settings

from src.cag.test_cag import test_cag
from src.rag.run_rag import run_rag

from src.rag.test_rag import test_rag
from src.cag.run_cag import run_cag

from config import CAG, RAG


async def main():
    load_dotenv()
    torch.serialization.add_safe_globals([DynamicCache])
    # torch.set_grad_enabled(False)

    if (CAG):
        cag_model_config = init_cag_settings()
        test_cag(cag_model_config)
        run_cag(cag_model_config)

    if (RAG):
        init_rag_settings()
        await test_rag()
        await run_rag()


if __name__ == "__main__":
    asyncio.run(main())
