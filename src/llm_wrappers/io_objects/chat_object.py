import json
from llm_wrappers.io_objects.base_io_object import BaseChatObject, IOModality

class TextChatObject(BaseChatObject):
    SUPPORTED_MODALITIES = [IOModality.TEXT]

    def __init__(self, sys_prompt:str):
        super().__init__(sys_prompt)

    def to_json(self)->str:
        # TODO: Add logging for metadata
        return json.dumps({
            'sys_prompt': self.sys_prompt,
            'history': [(p.to_json(), r.to_json()) for p, r in self._history]})

class MultiModalChatObject(BaseChatObject):
    SUPPORTED_MODALITIES = [
        IOModality.TEXT, IOModality.IMAGE, 
        IOModality.AUDIO, IOModality.VIDEO]

    def __init__(self, sys_prompt:str):
        super().__init__(sys_prompt)

    def to_json(self)->str:
        raise NotImplementedError('MultiModalChatObject.to_json() not implemented yet')
