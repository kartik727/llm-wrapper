"""Configuration for creating an LLM wrapper."""


class LLMConfig:
    """Configuration for creating an LLM wrapper."""

    def __init__(self, api_key: str = None, **kwargs):
        """Initializes the LLM configuration.

        Args:
            api_key (str, optional): The API key for the LLM. Defaults to None.
        """
        api_keys = {}
        if api_key is not None:
            api_keys['default'] = api_key

        for key, value in kwargs.items():
            if key.endswith('_api_key'):
                api_keys[key] = value

        self.api_keys = api_keys
        self.kwargs = kwargs
        self.model_kwargs = kwargs.get('model_kwargs', {})
