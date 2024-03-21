

from .base_wrapper import BaseWrapper
from .dummy_wrapper import DummyChatWrapper
from .openai_wrapper import OpenAIChatWrapper
from .anthropic_wrapper import AnthropicChatWrapper

from ..utils.config import LLMConfig

classes = {
    'BaseWrapper': BaseWrapper,
    'DummyChatWrapper': DummyChatWrapper,
    'OpenAIChatWrapper': OpenAIChatWrapper,
    'AnthropicChatWrapper': AnthropicChatWrapper,
}


class LLMWrapper:
    """Factory for creating chat wrappers."""

    @staticmethod
    def create(wrapper: str, api_key: str = None, **kwargs) -> BaseWrapper:
        """Creates a chat wrapper.

        Args:
            wrapper (str):
                The name of the wrapper.
            api_key (str, optional):
                The API key for the wrapper. Defaults to None.

        Returns:
            BaseWrapper: The chat wrapper.
        """
        config = LLMConfig(api_key, **kwargs)
        return classes[wrapper](config)
