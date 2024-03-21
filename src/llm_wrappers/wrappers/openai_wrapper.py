"""Wrapper for the OpenAI API."""

import json
from openai import OpenAI

from .base_wrapper import ChatWrapper
from ..utils.config import LLMConfig
from ..utils.io import ChatInput, ChatOutput
from ..utils.constants import modalities


class OpenAIChatWrapper(ChatWrapper):
    """Chat wrapper for the OpenAI API.
    Implements the `ChatWrapper` interface."""

    SUPPORTED_MODALITIES = [modalities.TEXT, modalities.TOOL_CALL]

    def __init__(self, config: LLMConfig):
        """Initializes the OpenAIChatWrapper.
        Args:
            config (LLMConfig): The config for the wrapper.
        """
        super().__init__(config)

        if self._config.api_keys.get('openai_api_key') is not None:
            self.api_key = self._config.api_keys['openai_api_key']

        self.client = OpenAI(api_key=self.api_key)
        self.model = self._config.kwargs['model']
        self.tool_handler = config.kwargs.get('tool_handler')

    def get_response(self, input: ChatInput, **kwargs) -> ChatOutput:
        while True:
            messages = []
            for (content, modality, metadata) in input.items:
                if modality in self.SUPPORTED_MODALITIES:
                    messages.append({'content': content, **metadata})
                else:
                    raise ValueError(f'Unsupported modality: {modality}')

            response_message = self.client.chat.completions.create(
                model=self.model, messages=messages,
                **kwargs).choices[0].message

            if self._is_tool_call(response_message):
                responses = self._handle_tool_call(response_message)
                input = ChatOutput(input, None, modality=modalities.TOOL_CALL,
                                   role='assistant',
                                   tool_calls=response_message.tool_calls)
                for (tool_id, func_name, response) in responses:
                    input = ChatOutput(input, response,
                                       modality=modalities.TOOL_CALL,
                                       role='tool', tool_call_id=tool_id,
                                       name=func_name)
            else:
                break

        return ChatOutput(input, response_message.content)

    def _is_tool_call(self, model_response) -> bool:
        # print(model_response.tool_calls)
        # return False
        return model_response.tool_calls is not None

    def _handle_tool_call(self, model_response) -> list[tuple[str, str, str]]:
        responses = []
        for tool_call in model_response.tool_calls:
            func_result = getattr(self.tool_handler, tool_call.function.name)(
                **json.loads(tool_call.function.arguments))
            responses.append((tool_call.id, tool_call.function.name,
                              func_result))
        return responses
