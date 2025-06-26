import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from accelerate import Accelerator

from .environ import HF_TOKEN
from config import MODEL_NAME


class ModelConfiguration:
    _accelerator = Accelerator()

    def __init__(self, model_name: str = MODEL_NAME):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=True
        )
        # self._model.eval()
        # self._model.to(self._accelerator.device)

    def model(self):
        return self._model

    def tokenizer(self):
        return self._tokenizer

    @classmethod
    def torch_device(cls) -> torch.device:
        return cls._accelerator.device
