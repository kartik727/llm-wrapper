import logging
import time

import openai
from openai.error import ServiceUnavailableError

from llm_wrappers.wrappers.base_wrapper import ChatLLMWrapper
from llm_wrappers.llm_config.openai_config import OpenAIConfig
from llm_wrappers.io_objects.openai_io_object import (Role, OpenAIMessage,
    OpenAIFunctionCall, UnknownResponseError, OpenAIChatObject)

class OpenAIWrapper(ChatLLMWrapper):
    def __init__(self, config:OpenAIConfig):
        super().__init__(config)

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
        
    def new_chat(self, sys_prompt:str)->OpenAIChatObject:
        return OpenAIChatObject(
            OpenAIMessage(
                Role.SYSTEM,
                sys_prompt)
            )

    def chat(self, context:OpenAIChatObject, user_prompt:str
        )->tuple[OpenAIChatObject, str]:
        context, response = super().chat(
            context,
            OpenAIMessage(Role.USER, user_prompt)
        )
        return context, response.text