import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_wrappers.base_wrapper import CompletionLLMWrapper, ChatLLMWrapper
from llm_wrappers.llama_wrapper.config import LlamaConfig
from llm_wrappers.llama_wrapper.helpers import LlamaMessage, Role

class LlamaWrapper(CompletionLLMWrapper, ChatLLMWrapper):
    INST_START = '[INST]'
    INST_END = '[/INST]'

    def __init__(self, config:LlamaConfig):
        super(LlamaWrapper, self).__init__(config)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_name,
            token = self._config.hf_api_key,
            **self._config.model_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            token = self._config.hf_api_key,
            **self._config.tokenizer_kwargs)
        self._model.to(self._config.device)

        if self._config.tokenizer_padding_token is not None:
            self._tokenizer.pad_token = self._config.tokenizer_padding_token

    def get_response(self, prompt:list[dict], **kwargs):
        templated_prompt = self._template_prompt(prompt)
        inputs = self._tokenizer(templated_prompt, return_tensors='pt').to(self._config.device)
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
        ).to(self._config.device)
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

