import json
from llm_wrappers.io_objects.base_io_object import BaseCompletionObject, IOModality

class TextCompletionObject(BaseCompletionObject):
    SUPPORTED_MODALITIES = [IOModality.TEXT]

    def to_json(self)->str:
        return json.dumps({
            'sys_prompt': self.sys_prompt,
            'completion': self._completion})

class MultiModalCompletionObject(BaseCompletionObject):
    SUPPORTED_MODALITIES = [
        IOModality.TEXT, IOModality.IMAGE, 
        IOModality.AUDIO, IOModality.VIDEO]

    def to_json(self)->str:
        raise NotImplementedError(
            'MultiModalCompletionObject.to_json() not implemented yet')
