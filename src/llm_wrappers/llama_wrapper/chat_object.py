from llm_wrappers.utils.chat_object import BaseChatObject, BaseMessage

class OpenAIChatObject(BaseChatObject):
    def formatted_prompt(self, prompt: BaseMessage) -> list[dict]:
        context = [self.sys_prompt.formatted_msg]
        for exchange in self._history:
            context.append(exchange[0].formatted_msg)
            context.append(exchange[1].formatted_msg)
        context.append(prompt.formatted_msg)
        return context

