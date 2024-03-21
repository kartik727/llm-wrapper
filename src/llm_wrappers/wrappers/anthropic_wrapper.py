"""Wrapper for the OpenAI API."""

from anthropic import Anthropic

from .base_wrapper import ChatWrapper
from ..utils.config import LLMConfig
from ..utils.io import ChatInput, ChatOutput
from ..utils.constants import modalities


class AnthropicChatWrapper(ChatWrapper):
    """Chat wrapper for the OpenAI API.
    Implements the `ChatWrapper` interface."""

    def __init__(self, config: LLMConfig):
        """Initializes the OpenAIChatWrapper.
        Args:
            config (LLMConfig): The config for the wrapper.
        """
        super().__init__(config)

        if self._config.api_keys.get('anthropic_api_key') is not None:
            self.api_key = self._config.api_keys['anthropic_api_key']

        self.client = Anthropic(api_key=self.api_key)
        self.model = self._config.kwargs['model']

    def get_response(self, input: ChatInput) -> ChatOutput:
        messages = []
        for (content, modality, metadata) in input.items:
            if modality == modalities.TEXT:
                messages.append({'role': metadata['role'], 'content': content})
            else:
                raise ValueError(f'Unsupported modality: {modality}')

        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            **self._config.model_kwargs).content[0].text

        return ChatOutput(input, response)
