from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Any

class BaseRole(Enum):
    ...

@dataclass
class BaseMessage(ABC):

    @property
    @abstractmethod
    def formatted_msg(self)->Any:
        ...

class ChatInfo:
    def __init__(self):
        self.stats = {
            'n_prompt_tokens': 0,
            'n_completion_tokens': 0,
            'n_total_tokens': 0
        }
        self.response_logs = []

class BaseChatObject(ABC):
    def __init__(self, sys_prompt:BaseMessage):
        self._sys_prompt = sys_prompt
        self._history = []

    @abstractmethod
    def formatted_prompt(self, prompt:BaseMessage)->Any:
        ...

    @property
    def sys_prompt(self)->BaseMessage:
        return self._sys_prompt

    @property
    def chat_length(self)->int:
        return len(self._history)

    def add_exchange(self, prompt:BaseMessage, response:BaseMessage):
        self._history.append((prompt, response))

    def reset(self)->None:
        self._history = []

class BaseCompletionObject(ABC):
    def __init__(self, sys_prompt:str):
        self._sys_prompt = sys_prompt

    @abstractmethod
    def formatted_prompt(self, prompt:BaseMessage)->Any:
        ...

    @property
    def sys_prompt(self)->str:
        return self._sys_prompt
