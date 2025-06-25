import torch

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow

from src.cag.cag import get_answer, clean_up
from src.cag.CAGModelManager import CAGModelManager
from src.rag.RAGAgentManager import RAGAgentManager
from params import CACHE_PATH



delimiter = "\n"



questions = [
        "Can you tell me what the context is about?",
        "What city does My best friend live in?",
        "What color are My best friend's eyes?",
        "What are My best friend's favorite writers?",
        "What is My best friend's cat called?",
        "Tell me about my best friend's family.",
        "What are the hobbies of my best friend's wife?",
        "What year was my best friend's wife born?"
    ]



def test_cag() -> None:
    model = CAGModelManager.get_model()
    tokenizer = CAGModelManager.get_tokenizer()
    device = CAGModelManager.get_device()

    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up(loaded_cache)



async def test_rag() -> None:
    agent: AgentWorkflow = await RAGAgentManager.get_rag_agent()
    test_context = Context(agent)

    for question in questions:
        print("Q:", question)
        answer = await agent.run(question, ctx=test_context)
        print("RAG:", answer)
        print(delimiter)
