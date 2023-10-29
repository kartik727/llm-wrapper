import torch

from llm_wrappers.utils.llm_config import BaseConfig

class LlamaConfig(BaseConfig):
    def __init__(self, model_name:str, hf_api_key:str, /, *,
            device:str='auto', tokenizer_padding_token:str=None, 
            model_kwargs:dict=None, tokenizer_kwargs:dict=None):
        self._model_name = model_name
        self._hf_api_key = hf_api_key
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device = device
        self._tokenizer_padding_token = tokenizer_padding_token
        self._model_kwargs = model_kwargs
        self._tokenizer_kwargs = tokenizer_kwargs

    @property
    def model_name(self)->str:
        return self._model_name

    @property
    def hf_api_key(self)->str:
        return self._hf_api_key

    @property
    def device(self)->str:
        return self._device

    @property
    def tokenizer_padding_token(self)->str:
        return self._tokenizer_padding_token

    @property
    def model_kwargs(self)->dict:
        return self._model_kwargs

    @property
    def tokenizer_kwargs(self)->dict:
        return self._tokenizer_kwargs

