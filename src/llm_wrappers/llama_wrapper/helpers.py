from enum import Enum
from dataclasses import dataclass

from llm_wrappers.utils.chat_object import BaseMessage

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
