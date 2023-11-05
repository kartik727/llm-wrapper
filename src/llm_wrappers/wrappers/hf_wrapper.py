import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_wrappers.wrappers.base_wrapper import CompletionLLMWrapper, ChatLLMWrapper
from llm_wrappers.llm_config import HFConfig
from llm_wrappers.io_objects.hf_io_object import (HFMessage, Role,
    HFChatObject, HFCompletionObject)

class HFWrapper(CompletionLLMWrapper, ChatLLMWrapper):
    def __init__(self, config:HFConfig):
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

    def get_response(self, prompt:list[dict], **kwargs)->HFMessage:
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
    
    def get_batch_response(self, prompts:list[list[dict]], batch_size:int=0, 
            **kwargs)->list[HFMessage]:
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
    
    def _parse_response(self, response:str)->HFMessage:
        return HFMessage(Role.ASSISTANT, response.strip())
    
    def _template_prompt(self, prompt:list[dict]):
        return self._tokenizer.apply_chat_template(prompt, tokenize=False)
    
    def _set_tokenizer_padding_token(self, padding_token:str):
        if padding_token is not None:
            self._tokenizer.pad_token = getattr(self._tokenizer, padding_token)
    
    def new_chat(self, sys_prompt:str, **kwargs)->HFChatObject:
        return HFChatObject(
            HFMessage(
                Role.SYSTEM,
                sys_prompt),
            **kwargs
            )
    
    def new_completion(self, sys_prompt:str, **kwargs)->HFCompletionObject:
        return HFCompletionObject(
            HFMessage(
                Role.SYSTEM,
                sys_prompt),
            **kwargs
            )
    
    def chat(self, context:HFChatObject, user_prompt:str
        )->tuple[HFChatObject, str]:
        context, response = super().chat(
            context,
            HFMessage(Role.USER, user_prompt),
            **context.chat_kwargs
        )
        return context, response.text
    
    def completion(self, comp_obj:HFCompletionObject, prompt:str)->str:
        return super().completion(
            comp_obj,
            HFMessage(Role.USER, prompt),
            **comp_obj.completion_kwargs
        ).text
    
    def batch_completion(self, comp_obj:HFCompletionObject,
            prompts:list[str], batch_size:int
        )->list[str]:
        completion = super().batch_completion(
            comp_obj,
            [HFMessage(Role.USER, prompt) for prompt in prompts],
            batch_size, **comp_obj.completion_kwargs)
        return [response.text for response in completion]

class LlamaWrapper(HFWrapper):
    INST_START = '[INST]'
    INST_END = '[/INST]'

    def _parse_response(self, response:str)->HFMessage:
        r_split = response.split(self.INST_END)
        return HFMessage(Role.ASSISTANT, r_split[-1].strip())
    
class ZephyrWrapper(HFWrapper):
    ASST_SEP = '<|assistant|>'

    def _parse_response(self, response:str)->HFMessage:
        r_split = response.split(self.ASST_SEP)
        return HFMessage(Role.ASSISTANT, r_split[-1].strip())