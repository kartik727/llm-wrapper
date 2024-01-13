from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class BaseRole(Enum):
    ...

class BaseMessage(ABC):
    def __init__(self, role:BaseRole, message):
        self._role = role
        self._message = message

class ChatInfo:
    def __init__(self):
        self.stats = {
            'n_prompt_tokens': 0,
            'n_completion_tokens': 0,
            'n_total_tokens': 0}
        self.response_logs = []

class IOModality(Enum):
    TEXT = 0
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3

class BaseIOObject(ABC):
    SUPPORTED_MODALITIES = []

    @abstractmethod
    def __init__(self, sys_prompt:BaseMessage):
        ...

    @abstractmethod
    def to_json(self)->str:
        ...

class BaseChatObject(BaseIOObject):
    def __init__(self, sys_prompt:BaseMessage):
        self._sys_prompt = sys_prompt
        self._history = []

    @property
    def sys_prompt(self)->BaseMessage:
        return self._sys_prompt
    
    @property
    def history(self)->list[tuple[BaseMessage, BaseMessage]]:
        return self._history

    @property
    def chat_length(self)->int:
        return len(self._history)

    def add_exchange(self, prompt:BaseMessage, response:BaseMessage):
        self._history.append((prompt, response))

    def reset(self)->None:
        self._history = []

class BaseCompletionObject(BaseIOObject):
    def __init__(self, sys_prompt:str):
        self._sys_prompt = sys_prompt
        self._completion = None

    @property
    def sys_prompt(self)->str:
        return self._sys_prompt
