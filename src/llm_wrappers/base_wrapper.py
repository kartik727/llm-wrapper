from abc import ABC, abstractmethod

from llm_wrappers.utils.chat_object import BaseChatObject, BaseSysObject, BaseMessage
from llm_wrappers.utils.llm_config import BaseConfig

class BaseLLMWrapper(ABC):
    def __init__(self, config:BaseConfig):
        self._config = config

    @abstractmethod
    def get_response(self, prompt)->BaseMessage:
        ...

    @abstractmethod
    def get_batch_response(self, prompts:list, batch_size:int)->list[BaseMessage]:
        ...

class CompletionLLMWrapper(BaseLLMWrapper):
    def completion(self, sys_prompt:BaseSysObject, prompt:BaseMessage)->BaseMessage:
        return self.get_response(sys_prompt.formatted_prompt(prompt))

    def batch_completion(self,
            sys_prompt:BaseSysObject,
            prompts:list[str|BaseChatObject],
            batch_size:int
        )->list[BaseMessage]:

        return self.get_batch_response(
            [sys_prompt.formatted_prompt(prompt) for prompt in prompts],
            batch_size)

class ChatLLMWrapper(BaseLLMWrapper):
    def chat(self,
            context:BaseChatObject,
            prompt:BaseMessage
        )->tuple[BaseChatObject, BaseMessage]:
        response = self.get_response(context.formatted_prompt(prompt))
        context.add_exchange(prompt, response)
        return context, response
