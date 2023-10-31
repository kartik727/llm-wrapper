from enum import Enum
from dataclasses import dataclass

from llm_wrappers.io_objects import BaseChatObject, BaseMessage

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

class OpenAIChatObject(BaseChatObject):
    def formatted_prompt(self, prompt: BaseMessage) -> list[dict]:
        context = [self.sys_prompt.formatted_msg]
        for exchange in self._history:
            context.append(exchange[0].formatted_msg)
            context.append(exchange[1].formatted_msg)
        context.append(prompt.formatted_msg)
        return context
