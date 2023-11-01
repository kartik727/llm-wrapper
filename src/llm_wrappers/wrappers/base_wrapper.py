from abc import ABC, abstractmethod

from llm_wrappers.io_objects.base_io_object import BaseChatObject, BaseCompletionObject, BaseMessage
from llm_wrappers.llm_config.base_config import BaseConfig

class BaseLLMWrapper(ABC):
    def __init__(self, config:BaseConfig):
        self._config = config

    @abstractmethod
    def get_response(self, prompt, **kwargs)->BaseMessage:
        ...

    @abstractmethod
    def get_batch_response(self, prompts:list, batch_size:int, **kwargs)->list[BaseMessage]:
        ...

class CompletionLLMWrapper(BaseLLMWrapper):
    def completion(self, comp_obj:BaseCompletionObject, prompt:BaseMessage,
            **kwargs)->BaseMessage:
        return self.get_response(
            comp_obj.formatted_prompt(prompt, **kwargs))

    def batch_completion(self,
            comp_obj:BaseCompletionObject,
            prompts:list[str|BaseChatObject],
            batch_size:int,
            **kwargs
        )->list[BaseMessage]:

        return self.get_batch_response(
            [comp_obj.formatted_prompt(prompt) for prompt in prompts],
            batch_size, **kwargs)

class ChatLLMWrapper(BaseLLMWrapper):
    def chat(self,
            context:BaseChatObject,
            prompt:BaseMessage,
            **kwargs
        )->tuple[BaseChatObject, BaseMessage]:
        response = self.get_response(context.formatted_prompt(prompt), **kwargs)
        context.add_exchange(prompt, response)
        return context, response
