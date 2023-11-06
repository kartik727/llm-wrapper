"""Base class and interfaces for LLM wrappers.

Wrappers are used to abstract away the details of interacting with the API.
They should implement the following methods:
    
    get_response(prompt, **kwargs)
    get_batch_response(prompts, **kwargs)

Optionally, they can implement the following methods if they subclass
`CompletionLLMWrapper` or `ChatLLMWrapper`:
    
    new_chat(sys_prompt, **kwargs)
    chat(context, prompt, **kwargs)

    new_completion(sys_prompt, **kwargs)
    completion(context, prompt, **kwargs)
"""

from abc import ABC, abstractmethod

from llm_wrappers.io_objects.base_io_object import (
    BaseChatObject, BaseCompletionObject, BaseMessage)
from llm_wrappers.llm_config.base_config import BaseConfig

class BaseLLMWrapper(ABC):
    """Base class for LLM wrappers.
    """

    def __init__(self, config:BaseConfig):
        """Initializes the LLM wrapper.""" 
        self._config = config

    @abstractmethod
    def get_response(self, prompt, **kwargs)->BaseMessage:
        """Sends a user prompt to the LLM and returns the response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            BaseMessage: The response from the LLM.
        """

    @abstractmethod
    def get_batch_response(self, prompts:list, **kwargs)->list[BaseMessage]:
        """Sends a batch of user prompts to the LLM and returns the responses.

        Args:
            prompts (list): The list of prompts to send to the LLM.

        Returns:
            list[BaseMessage]: The list of responses from the LLM.
        """

class CompletionLLMWrapper(BaseLLMWrapper):
    """Base class for LLM wrappers that implement completion. The intended
    use case is to get one response from the LLM after giving it one prompt,
    and optionally setting a system prompt beforehand.
    """

    @abstractmethod
    def new_completion(self, sys_prompt:str, **kwargs)->BaseCompletionObject:
        """Creates a new completion object.

        Args:
            sys_prompt (str): The system prompt to use for the completion.

        Returns:
            BaseCompletionObject: The completion object.
        """

    def completion(self, comp_obj:BaseCompletionObject, prompt:BaseMessage,
            **kwargs)->BaseMessage:
        """Gets a response from the LLM.

        Args:
            comp_obj (BaseCompletionObject): The completion object.
            prompt (BaseMessage): The user prompt.

        Returns:
            BaseMessage: The response from the LLM.
        """
        return self.get_response(
            comp_obj.formatted_prompt(prompt, **kwargs))

    def batch_completion(self,
            comp_obj:BaseCompletionObject,
            prompts:list[str|BaseChatObject],
            **kwargs
        )->list[BaseMessage]:
        """Gets a batch of responses from the LLM.

        Args:
            comp_obj (BaseCompletionObject): The completion object.
            prompts (list[str | BaseChatObject]): The list of user prompts.

        Returns:
            list[BaseMessage]: The list of responses from the LLM.
        """
        return self.get_batch_response(
            [comp_obj.formatted_prompt(prompt) for prompt in prompts],
            **kwargs)

class ChatLLMWrapper(BaseLLMWrapper):
    """Base class for LLM wrappers that implement chat. The intended use case
    is to get a sequence of alternating responses from the LLM and user in 
    a chat-like setting, and optionally setting a system prompt beforehand.
    """

    @abstractmethod
    def new_chat(self, sys_prompt:str, /, **kwargs)->BaseChatObject:
        """Creates a new chat object.

        Args:
            sys_prompt (str): The system prompt to use for the chat.

        Returns:
            BaseChatObject: The chat object.
        """

    def chat(self,
            context:BaseChatObject,
            prompt:BaseMessage,
            /,
            **kwargs
        )->tuple[BaseChatObject, BaseMessage]:
        """Sends a user prompt to the LLM and returns the response. The
        intended use case is to call this method in a loop, with the response
        from the previous call as the `context` argument.

        Args:
            context (BaseChatObject): The chat object.
            prompt (BaseMessage): The user prompt.

        Returns:
            tuple[BaseChatObject, BaseMessage]: The updated chat context and
                the response from the LLM.
        """
        response = self.get_response(context.formatted_prompt(prompt), **kwargs)
        context.add_exchange(prompt, response)
        return context, response
