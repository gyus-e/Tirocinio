import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from environ import HF_TOKEN
from config import (
    MODEL_NAME,
    TEMPERATURE,
    CONTEXT_WINDOW
)


class ModelManager:
    _model = None
    _tokenizer = None


    @classmethod
    def get_model(cls, model_name: str = MODEL_NAME):
        if cls._model is None:
            cls._load_model(model_name)
        return cls._model
    

    @classmethod
    def get_tokenizer(cls, model_name: str = MODEL_NAME):
        if cls._tokenizer is None:
            cls._load_tokenizer(model_name)
        return cls._tokenizer
    

    # If you've set type checking in your IDE, this method could signal false positives
    @classmethod
    def get_torch_device(cls) -> torch.device:
        # Mistral/Llama models
        if hasattr(cls._model, 'model') and hasattr(cls._model.model, 'embed_tokens'):
            device = cls._model.model.embed_tokens.weight.device
        
        # GPT-2 models
        elif hasattr(cls._model, 'transformer') and hasattr(cls._model.transformer, 'wte'):
            device = cls._model.transformer.wte.weight.device
        
        # Fallback to first parameter device
        else:
            device = next(cls._model.parameters()).device

        return device


    @classmethod
    def _load_model(cls, model_name: str = MODEL_NAME):

        cls._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )

        cls._model.eval()


    @classmethod
    def _load_tokenizer(cls, model_name: str = MODEL_NAME):
        
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=HF_TOKEN, 
            trust_remote_code=True
        )