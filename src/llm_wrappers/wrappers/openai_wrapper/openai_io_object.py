from enum import Enum
from dataclasses import dataclass

from llm_wrappers.io_objects import BaseChatObject, BaseMessage

class UnknownResponseError(Exception):
    pass

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    FUNCTION = 'tool'

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
    tool_call_id: str
    text: str

    @property
    def formatted_msg(self):
        return {'tool_call_id' : self.tool_call_id, 'role' : self.role.value, 
                'name' : self.name, 'content' : self.text}

@dataclass
class OpenAIFunctionCall(BaseMessage):
    role: Role
    name: str
    tool_call_id: str
    params: dict

    @property
    def formatted_msg(self):
        return { 'content' : None, 'role' : self.role.value,
            'tool_calls' : [
                {'id' : self.tool_call_id, 'function' : {
                    'name' : self.name, 'arguments' : self.params},
                'type' : 'function'}]}

class OpenAIChatObject(BaseChatObject):
    def to_json(self) -> str:
        raise NotImplementedError('OpenAIChatObject.to_json() not implemented yet')
