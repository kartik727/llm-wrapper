"""OpenAI LLM Wrapper

This module contains a wrapper for the OpenAI API.
"""

import logging
import time
from typing import Any

from openai import OpenAI, RateLimitError

from llm_wrappers.wrappers.base_wrapper import ChatLLMWrapper
from llm_wrappers.llm_config.openai_config import OpenAIConfig
from llm_wrappers.io_objects.base_io_object import BaseChatObject, BaseMessage
from .openai_io_object import (Role, OpenAIMessage, OpenAIFunctionCall,
    OpenAIFunctionResponse, UnknownResponseError, OpenAIChatObject)
class OpenAIWrapper(ChatLLMWrapper):
    """Wrapper for the OpenAI API. Implements the ChatLLMWrapper interface.

    This wrapper can be used to query models like `gpt-3.5-turbo` and `gpt-4`
    from the OpenAI API.
    """
    def __init__(self, config:OpenAIConfig, /, *, functions:list[dict]=None):
        """Initializes the OpenAIWrapper.

        Args:
            config (OpenAIConfig): The config for the wrapper.
            functions (list[dict], optional): A list of dicts, each dict
                representing a function. Defaults to None.
        """
        super().__init__(config)
        self._client = OpenAI(api_key=config.api_key)

        self._chat_kwargs = {}
        if functions is not None:
            self._chat_kwargs['tools'] = [
                {
                    'type' : 'function',
                    'function' : func
                } for func in functions]

    def _formatted_msg(self, msg:BaseMessage)->dict:
        if isinstance(msg, OpenAIFunctionCall):
            return { 
                'role' : msg.role.value,
                'content' : None, 
                'tool_calls' : [{
                    'id' : msg.tool_call_id,
                    'type' : 'function',
                    'function' : {
                        'name' : msg.name,
                        'arguments' : msg.params}}]}

        elif isinstance(msg, OpenAIFunctionResponse):
            return {
                'tool_call_id' : msg.tool_call_id,
                'role' : msg.role.value, 
                'name' : msg.name,
                'content' : msg.text}

        else:
            return {'role' : msg.role.value, 'content' : msg.text}

    def formatted_prompt(self, context: BaseChatObject, prompt: BaseMessage) -> Any:
        formatted = [self._formatted_msg(context.sys_prompt)]
        for (p, r) in context.history:
            formatted.append(self._formatted_msg(p))
            formatted.append(self._formatted_msg(r))
        formatted.append(self._formatted_msg(prompt))
        return formatted

    def get_response(self, prompt:list[dict], **kwargs
        )->OpenAIMessage | OpenAIFunctionCall:
        """Makes a request to the OpenAI API with the user prompt and 
        returns the parsed response.

        Args:
            prompt (list[dict]): The prompt to send to the API. This should be
                a list of dicts, each dict representing a message.

        Returns:
            OpenAIMessage | OpenAIFunctionCall: The parsed response from the
                API.
        """
        # Get the response. Keep retrying until success
        success = False
        while not success:
            try:
                api_response = self._client.chat.completions.create(
                    model = self._config.model_name,
                    messages = prompt,
                    **self._chat_kwargs)

                ## TODO: Add stats to the chat object
                # self._n_prompt_tokens += api_response['usage']['prompt_tokens']
                # self._n_completion_tokens += api_response['usage']['completion_tokens']

                success = True
                # self.response_logs.append(ast_result)
            except RateLimitError:
                logging.warning('Rate limit reached. Retrying after %d s.',
                    self._config.retry_wait_time)
                time.sleep(self._config.retry_wait_time)

        return self._parse_api_response(api_response)

    def get_batch_response(self, prompts:list[list[dict]], **kwargs
        )->list[OpenAIMessage | OpenAIFunctionCall]:
        """Makes a batch request to the OpenAI API with the user prompts and
        returns the parsed responses. This is a convenience method for calling
        `get_response` multiple times.

        Args:
            prompts (list[list[dict]]): List of prompts to send to the API.

        Returns:
            list[OpenAIMessage | OpenAIFunctionCall]: List of parsed responses
                from the API.
        """
        return [self.get_response(prompt) for prompt in prompts]

    def _parse_api_response(self, api_response:dict
        )->OpenAIMessage | OpenAIFunctionCall:
        """Parses the response from the OpenAI API.

        Args:
            api_response (dict): The response from the API.

        Raises:
            UnknownResponseError: If the response cannot be parsed.

        Returns:
            OpenAIMessage | OpenAIFunctionCall: The parsed response.
        """
        api_result = api_response.choices[0]
        assert api_result.message.role == 'assistant', \
            ('Prompt response must be from `assistant`, '
            f'not `{api_result.message.role}`')

        logging.debug('Prompt response (raw): %s', api_result)

        if api_result.finish_reason == 'stop':
            api_txt = api_result.message.content
            return OpenAIMessage(Role.ASSISTANT, api_txt)
        elif api_result.finish_reason == 'tool_calls':
            tool_call_obj = api_result.message.tool_calls[0]
            return OpenAIFunctionCall(Role.ASSISTANT, tool_call_obj.function.name,
                tool_call_obj.id, tool_call_obj.function.arguments)
        elif api_result.finish_reason is None and api_result.finish_details.get('type') == 'max_tokens':
            api_txt = api_result.message.content
            logging.warning('Max tokens reached. Response might be incomplete.')
            return OpenAIMessage(Role.ASSISTANT, api_txt)
        else:
            raise UnknownResponseError('Messages with '
                f'`finish_reason`==`{api_result.finish_reason}` '
                'cannot be parsed yet.')

    def _handle_function_call(self, fc:OpenAIFunctionCall)->OpenAIFunctionResponse:
        """Handles a function call from the API. This method should be
        overridden by subclasses that support function calls.

        Args:
            fc (OpenAIFunctionCall): The function call to handle.

        Raises:
            NotImplementedError: If function calls are not supported by the
                wrapper.
        """
        raise NotImplementedError('Function calls need to be handled by a '+
            'subclass of OpenAIWrapper.')

    def new_chat(self, sys_prompt:str, /, **kwargs)->OpenAIChatObject:
        return OpenAIChatObject(
            OpenAIMessage(
                Role.SYSTEM,
                sys_prompt))

    def chat(self, context:OpenAIChatObject, user_prompt:str, /, **kwargs
        )->tuple[OpenAIChatObject, str]:
        """Sends a user prompt to the API and returns the response. The 
        intended use case is to call this method in a loop, with the response
        to be sent to the user while the updated context is passed back into
        the next iteration with the next user prompt.

        Args:
            context (OpenAIChatObject): The context of the conversation.
            user_prompt (str): The user prompt to send to the API.

        Raises:
            UnknownResponseError: If the response cannot be parsed.

        Returns:
            tuple[OpenAIChatObject, str]: The updated context and the response
                from the API.
        """
        message = OpenAIMessage(Role.USER, user_prompt)
        while True:
            context, response = super().chat(
                context,
                message
            )

            if isinstance(response, OpenAIMessage):
                return context, response.text

            elif isinstance(response, OpenAIFunctionCall):
                message = self._handle_function_call(response)

            else:
                raise UnknownResponseError('Response of type `'
                    +type(response)+'` cannot be parsed yet.')
