from enum import Enum
from dataclasses import dataclass

from llm_wrappers.io_objects import BaseChatObject, BaseCompletionObject, BaseMessage

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

@dataclass
class LlamaMessage(BaseMessage):
    role: Role
    text: str

    @property
    def formatted_msg(self):
        return {'role' : self.role.value, 'content' : self.text}

class LlamaChatObject(BaseChatObject):
    def formatted_prompt(self, prompt: BaseMessage) -> list[dict]:
        context = [self.sys_prompt.formatted_msg]
        for exchange in self._history:
            context.append(exchange[0].formatted_msg)
            context.append(exchange[1].formatted_msg)
        context.append(prompt.formatted_msg)
        return context

class LlamaCompletionObject(BaseCompletionObject):
    def formatted_prompt(self, prompt: BaseMessage) -> list[dict]:
        return [self.sys_prompt.formatted_msg, prompt.formatted_msg]