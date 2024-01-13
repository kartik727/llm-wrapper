# import logging
from enum import Enum
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_wrappers.wrappers.base_wrapper import CompletionLLMWrapper, ChatLLMWrapper
from llm_wrappers.io_objects.base_io_object import BaseIOObject, BaseMessage
from llm_wrappers.io_objects.chat_object import TextChatObject
from llm_wrappers.io_objects.completion_object import TextCompletionObject
from llm_wrappers.llm_config import HFConfig
# from llm_wrappers.io_objects.hf_io_object import (HFMessage, Role,
#     HFChatObject, HFCompletionObject)

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

class HFWrapper(CompletionLLMWrapper, ChatLLMWrapper):
    def __init__(self, config:HFConfig):
        super().__init__(config)

        if self._config.device == 'try_cuda':
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

    def get_response(self, prompt:list[dict], **kwargs)->BaseMessage:
        templated_prompt = self._template_prompt(prompt)
        inputs = self._tokenizer(templated_prompt, return_tensors='pt').to(self._device)
        generated_ids = self._model.generate(inputs.input_ids, **kwargs)
        response = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        info = {
            'input_length' : inputs.input_ids.shape,
            'output_length' : generated_ids.shape,
            'raw_response' : response}

        return self._parse_response(response)

    def get_batch_response(self, prompts:list[list[dict]], batch_size:int=0,
            **kwargs)->list[BaseMessage]:
        templated_prompts = [self._template_prompt(prompt) for prompt in prompts]
        inputs = self._tokenizer(
            templated_prompts, return_tensors='pt', padding=True
        ).to(self._device)
        generated_ids = self._model.generate(inputs.input_ids, **kwargs)

        info = {
            'input_length' : inputs.input_ids.shape,
            'output_length' : generated_ids.shape,
            'raw_responses' : []}

        responses = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        return [self._parse_response(response) for response in responses]

    @staticmethod
    def format_msg(msg:BaseMessage)->dict:
        return {'role' : msg._role.value, 'content' : msg._message}

    def formatted_prompt(self, context: BaseIOObject, prompt: BaseMessage)->list[dict]:
        context = [self.format_msg(context.sys_prompt)]
        for exchange in context._history:
            context.append(self.format_msg(exchange[0]))
            context.append(self.format_msg(exchange[1]))
        context.append(self.format_msg(prompt))
        return context

    def _parse_response(self, response:str)->BaseMessage:
        return BaseMessage(Role.ASSISTANT, response.strip())

    def _template_prompt(self, prompt:list[dict]):
        return self._tokenizer.apply_chat_template(prompt, tokenize=False)

    def _set_tokenizer_padding_token(self, padding_token:str):
        if padding_token is not None:
            self._tokenizer.pad_token = getattr(self._tokenizer, padding_token)

    def new_chat(self, sys_prompt:str, **kwargs)->TextChatObject:
        return TextChatObject(
            BaseMessage(
                Role.SYSTEM,
                sys_prompt),
            **kwargs)

    def new_completion(self, sys_prompt:str, **kwargs)->TextCompletionObject:
        return TextCompletionObject(
            BaseMessage(
                Role.SYSTEM,
                sys_prompt),
            **kwargs)

    def chat(self, context:TextChatObject, user_prompt:str
            )->tuple[TextChatObject, str]:
        context, response = super().chat(
            context,
            # HFMessage(Role.USER, user_prompt),
            BaseMessage(Role.USER, user_prompt))
        return context, response.text

    def completion(self, comp_obj:TextCompletionObject, prompt:str)->str:
        return super().completion(
            comp_obj,
            # HFMessage(Role.USER, prompt),
            BaseMessage(Role.USER, prompt),
            **comp_obj.completion_kwargs)._message

    def batch_completion(self, comp_obj:TextCompletionObject,
            prompts:list[str], batch_size:int
        )->list[str]:
        completion = super().batch_completion(
            comp_obj,
            [BaseMessage(Role.USER, prompt) for prompt in prompts],
            batch_size, **comp_obj.completion_kwargs)
        return [response._message for response in completion]

class LlamaWrapper(HFWrapper):
    INST_START = '[INST]'
    INST_END = '[/INST]'

    def _parse_response(self, response:str)->BaseMessage:
        r_split = response.split(self.INST_END)
        return BaseMessage(Role.ASSISTANT, r_split[-1].strip())

class ZephyrWrapper(HFWrapper):
    ASST_SEP = '<|assistant|>'

    def _parse_response(self, response:str)->BaseMessage:
        r_split = response.split(self.ASST_SEP)
        return BaseMessage(Role.ASSISTANT, r_split[-1].strip())
