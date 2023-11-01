import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_wrappers.wrappers.base_wrapper import CompletionLLMWrapper, ChatLLMWrapper
from llm_wrappers.llm_config import LlamaConfig
from llm_wrappers.io_objects.llama_io_object import (LlamaMessage, Role,
    LlamaChatObject, LlamaCompletionObject)

class LlamaWrapper(CompletionLLMWrapper, ChatLLMWrapper):
    INST_START = '[INST]'
    INST_END = '[/INST]'

    def __init__(self, config:LlamaConfig):
        super().__init__(config)

        if self._config.device in ['auto', 'try_cuda']:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = self._config.device

        if self._config.device == 'auto':
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name,
                token = self._config.hf_api_key,
                device_map = 'auto',
                **self._config.model_kwargs)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name,
                token = self._config.hf_api_key,
                **self._config.model_kwargs)
            self._model.to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            token = self._config.hf_api_key,
            **self._config.tokenizer_kwargs)

        self._set_tokenizer_padding_token(self._config.tokenizer_padding_token)

    def get_response(self, prompt:list[dict], **kwargs):
        templated_prompt = self._template_prompt(prompt)
        inputs = self._tokenizer(templated_prompt, return_tensors='pt').to(self._device)
        generated_ids = self._model.generate(inputs.input_ids, **kwargs)
        response = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        info = {
            'input_length' : inputs.input_ids.shape,
            'output_length' : generated_ids.shape,
            'raw_response' : response
        }

        return self._parse_response(response)
    
    def get_batch_response(self, prompts:list[list[dict]], batch_size:int=0, **kwargs):
        templated_prompts = [self._template_prompt(prompt) for prompt in prompts]
        inputs = self._tokenizer(
            templated_prompts, return_tensors='pt', padding=True
        ).to(self._device)
        generated_ids = self._model.generate(inputs.input_ids, **kwargs)
        
        info = {
            'input_length' : inputs.input_ids.shape,
            'output_length' : generated_ids.shape,
            'raw_responses' : []
        }

        responses = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return [self._parse_response(response) for response in responses]
    
    def _parse_response(self, response:str):
        r_split = response.split(self.INST_END)
        return LlamaMessage(Role.ASSISTANT, r_split[-1].strip())
    
    def _template_prompt(self, prompt:list[dict]):
        return self._tokenizer.apply_chat_template(prompt, tokenize=False)
    
    def _set_tokenizer_padding_token(self, padding_token:str):
        if padding_token is not None:
            self._tokenizer.pad_token = getattr(self._tokenizer, padding_token)
    
    def new_chat(self, sys_prompt:str)->LlamaChatObject:
        return LlamaChatObject(
            LlamaMessage(
                Role.SYSTEM,
                sys_prompt)
            )
    
    def new_completion(self, sys_prompt:str)->LlamaCompletionObject:
        return LlamaCompletionObject(
            LlamaMessage(
                Role.SYSTEM,
                sys_prompt)
            )
    
    def chat(self, context:LlamaChatObject, user_prompt:str
        )->tuple[LlamaChatObject, str]:
        context, response = super().chat(
            context,
            LlamaMessage(Role.USER, user_prompt),
            **context.chat_kwargs
        )
        return context, response.text
    
    def completion(self, comp_obj:LlamaCompletionObject, prompt:str)->str:
        return super().completion(
            comp_obj,
            LlamaMessage(Role.USER, prompt),
            **comp_obj.completion_kwargs
        ).text
    
    def batch_completion(self, comp_obj:LlamaCompletionObject,
            prompts:list[str], batch_size:int
        )->list[str]:
        completion = super().batch_completion(
            comp_obj,
            [LlamaMessage(Role.USER, prompt) for prompt in prompts],
            batch_size, **comp_obj.completion_kwargs)
        return [response.text for response in completion]
