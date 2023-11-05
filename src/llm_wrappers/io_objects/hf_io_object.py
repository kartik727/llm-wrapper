from enum import Enum
from dataclasses import dataclass

from llm_wrappers.io_objects import BaseChatObject, BaseCompletionObject, BaseMessage

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

@dataclass
class HFMessage(BaseMessage):
    role: Role
    text: str

    @property
    def formatted_msg(self):
        return {'role' : self.role.value, 'content' : self.text}

class HFChatObject(BaseChatObject):
    def __init__(self, sys_prompt:HFMessage, **kwargs):
        super().__init__(sys_prompt)
        self._chat_kwargs = kwargs

    def formatted_prompt(self, prompt: HFMessage) -> list[dict]:
        context = [self.sys_prompt.formatted_msg]
        for exchange in self._history:
            context.append(exchange[0].formatted_msg)
            context.append(exchange[1].formatted_msg)
        context.append(prompt.formatted_msg)
        return context
    
    @property
    def chat_kwargs(self)->dict:
        return self._chat_kwargs

class HFCompletionObject(BaseCompletionObject):
    def __init__(self, sys_prompt:HFMessage, **kwargs):
        super().__init__(sys_prompt)
        self._completion_kwargs = kwargs

    def formatted_prompt(self, prompt: HFMessage) -> list[dict]:
        return [self.sys_prompt.formatted_msg, prompt.formatted_msg]
    
    @property
    def completion_kwargs(self)->dict:
        return self._completion_kwargs
