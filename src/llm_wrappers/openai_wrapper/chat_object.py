from llm_wrappers.utils.chat_object import BaseChatObject, BaseMessage
from llm_wrappers.openai_wrapper.helpers import OpenAIMessage, Role

class OpenAIChatObject(BaseChatObject):
    def __init__(self, sys_prompt:str):
        super(OpenAIChatObject, self).__init__(OpenAIMessage(Role.SYSTEM, sys_prompt))

    def formatted_prompt(self, prompt: BaseMessage) -> list[dict]:
        context = [self.sys_prompt.formatted_msg]
        for exchange in self._history:
            context.append(exchange[0].formatted_msg)
            context.append(exchange[1].formatted_msg)
        context.append(prompt.formatted_msg)
        return context

