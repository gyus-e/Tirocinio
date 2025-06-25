
import asyncio
import torch
from transformers.cache_utils import DynamicCache
from src.rag.init_rag_settings import init_rag_settings
from src.cag.init_cag_settings import init_cag_settings
from test import test_cag, test_rag


async def main():
    torch.set_grad_enabled(False)
    torch.serialization.add_safe_globals([DynamicCache])
    init_cag_settings()
    test_cag()
    init_rag_settings()
    await test_rag()
    

if __name__ == "__main__":
    asyncio.run(main())
