import asyncio
import torch
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache
from config import DO_CAG, DO_RAG
from utils import get_models_list, get_embed_models_list


async def main():
    load_dotenv()
    torch.serialization.add_safe_globals([DynamicCache])
    # torch.set_grad_enabled(False)

    print(get_models_list())
    print(get_embed_models_list())

    if DO_RAG:
        import rag
        from rag.test_rag import test_rag
        from rag.run_rag import run_rag

        await test_rag()
        # await run_rag()

    if DO_CAG:
        import cag
        from cag.test_cag import test_cag
        from cag.run_cag import run_cag

        test_cag()
        # run_cag()


if __name__ == "__main__":
    asyncio.run(main())
