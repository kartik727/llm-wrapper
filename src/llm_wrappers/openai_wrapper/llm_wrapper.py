import logging
import time
import json
from dataclasses import dataclass
from enum import Enum
import shlex
import pandas as pd

import openai
from openai.error import ServiceUnavailableError

from llm_wrappers.base_wrapper import ChatLLMWrapper
from llm_wrappers.openai_wrapper.chat_object import OpenAIChatObject
from llm_wrappers.openai_wrapper.config import OpenAIConfig
from llm_wrappers.openai_wrapper.helpers import Role, OpenAIMessage, OpenAIFunctionCall, OpenAIFunctionResponse, UnknownResponseError

openai.api_key = '' # Enter you API key here
model_name = 'gpt-3.5-turbo'

logging.basicConfig(level=logging.DEBUG)

class OpenAIWrapper(ChatLLMWrapper):
    def __init__(self, config:OpenAIConfig):
        super(OpenAIWrapper, self).__init__(config)

    def get_response(self, prompt:list[dict]):
        # Get the response. Keep retrying until success
        openai.api_key = self._config.api_key
        success = False
        while not success:
            try:
                api_response = openai.ChatCompletion.create(
                    model = self._config.model_name,
                    messages = prompt)
                    ## TODO: Add chat kwargs for functions
                    # **self.chat_kwargs)
                
                ## TODO: Add stats to the chat object
                # self._n_prompt_tokens += api_response['usage']['prompt_tokens']
                # self._n_completion_tokens += api_response['usage']['completion_tokens']

                success = True
                # self.response_logs.append(ast_result)
            except ServiceUnavailableError:
                logging.warning(f'Service not available. Retrying after {self._config.retry_wait_time} s.')
                time.sleep(self._config.retry_wait_time)

        return self._parse_api_response(api_response)
    
    def get_batch_response(self, prompts:list[list[dict]], batch_size:int):
        return [self.get_response(prompt) for prompt in prompts]
        
    def _parse_api_response(self, api_response:dict):
        # Parse the response
        api_result = api_response['choices'][0]
        assert api_result['message']['role'] == 'assistant', \
            f'Prompt response must be from `assistant`, not `{api_result["message"]["role"]}`'

        ## TODO: Parse the response
        if api_result['finish_reason'] == 'stop':
            api_txt = api_result['message']['content']
            logging.debug(f'Prompt response (text): {api_txt}')
            return OpenAIMessage(Role.ASSISTANT, api_txt)
        elif api_result['finish_reason'] == 'function_call':
            api_fc_params = api_result['message']['function_call']
            logging.debug(f'Prompt response (func call): {api_fc_params}')
            return OpenAIFunctionCall(Role.ASSISTANT, api_fc_params)
        else:
            raise UnknownResponseError(f'Messages with `finish_reason`==`{api_result["finish_reason"]}` cannot be parsed yet.')
