"""Dummy wrapper for testing purposes."""

from .base_wrapper import ChatWrapper
from ..utils.io import ChatInput, ChatOutput


class DummyChatWrapper(ChatWrapper):
    def get_response(self, input: ChatInput) -> ChatOutput:
        text = input.prompt.upper()
        return ChatOutput(input, text)
