from llm_wrappers.llm_config.base_config import BaseConfig

class OpenAIConfig(BaseConfig):
    def __init__(self, model_name:str, api_key:str, /, *,
            retry_wait_time:float=5):
        self._model_name = model_name
        self._api_key = api_key
        self._retry_wait_time = retry_wait_time

    @property
    def model_name(self)->str:
        return self._model_name

    @property
    def api_key(self)->str:
        return self._api_key

    @property
    def retry_wait_time(self)->float:
        return self._retry_wait_time
