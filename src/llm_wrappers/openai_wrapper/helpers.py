from enum import Enum
from dataclasses import dataclass

from llm_wrappers.utils.chat_object import BaseMessage

class UnknownResponseError(Exception):
    pass

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    FUNCTION = 'function'

@dataclass
class OpenAIMessage(BaseMessage):
    role: Role
    text: str
    
    @property
    def formatted_msg(self):
        return {'role' : self.role.value, 'content' : self.text}


@dataclass
class OpenAIFunctionResponse(BaseMessage):
    role: Role
    name: str
    text: str

    @property
    def formatted_msg(self):
        return {'role' : self.role.value, 'name' : self.name, 'content' : self.text}

@dataclass
class OpenAIFunctionCall(BaseMessage):
    role: Role
    params: dict

    @property
    def formatted_msg(self):
        return {'role' : self.role.value, 'content' : None, 'function_call' : self.params}