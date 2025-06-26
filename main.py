import asyncio
import torch
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache

from src.cag.init_cag_settings import init_cag_settings
from src.cag.test_cag import test_cag

from src.rag.init_rag_settings import init_rag_settings
from src.rag.test_rag import test_rag


async def main():
    load_dotenv()

    torch.set_grad_enabled(False)
    torch.serialization.add_safe_globals([DynamicCache])

    init_rag_settings()
    await test_rag()
    
    # init_cag_settings()
    # test_cag()
    
    

if __name__ == "__main__":
    asyncio.run(main())
