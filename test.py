import asyncio

async def test():
    from rag.test_rag import test_rag
    await test_rag()

    from cag.test_cag import test_cag
    test_cag()

if __name__ == "__main__":
    asyncio.run(test())