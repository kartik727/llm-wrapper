from abc import ABC, abstractmethod

from ..utils.config import LLMConfig
from ..utils.io import Input, BaseIO, ChatOutput


class BaseWrapper(ABC):
    def __init__(self, config: LLMConfig, **kwargs):
        self._config = config
        self.api_key = config.api_keys.get('default')

    @abstractmethod
    def get_response(self, input: Input) -> BaseIO:
        """Sends the user input to the LLM and returns the response.

        Args:
            input (Input): The user input.

        Returns:
            BaseIO: The response from the LLM.
        """

    def get_batch_response(self, inputs: list[Input]) -> list[BaseIO]:
        """Sends a batch of user inputs to the LLM and returns the responses.

        Args:
            inputs (list[Input]): The list of user inputs.

        Returns:
            list[BaseIO]: The list of responses from the LLM.
        """
        return [self.get_response(input) for input in inputs]


class ChatWrapper(BaseWrapper):
    def chat(self, prompt: str, *, context: list[tuple] | None = None,
             sys_prompt: str | None = None, **kwargs) -> ChatOutput:
        """Creates a chat input object and sends it to the LLM.

        Args:
            prompt (str): The user prompt.
            context (list[tuple], optional): The context. Defaults to None.
            sys_prompt (str, optional): The system prompt. Defaults to None.

        Returns:
            BaseIO: The response from the LLM.
        """
        if not ((context is None) or (sys_prompt is None)):
            raise ValueError('Can not set both context and system prompt')

        if context is not None:
            if len(context) == 0:
                pass
            elif context[0][2]['role'] == 'system':
                sys_prompt = context[0][0]
                history = [(c[0], c[1]) for c in context[1:]]
            else:
                history = [(c[0], c[1]) for c in context]
        else:
            history = []

        input = Input.chat(prompt, history=history, sys_prompt=sys_prompt)

        return self.get_response(input, **kwargs)

    def _is_tool_call(self, model_response) -> bool:
        """Check if the model response is a tool call."""
        return False

    def _handle_tool_call(self, model_response) -> list[str]:
        raise NotImplementedError('This method should be implemented in a '
                                  'subclass of ChatWrapper')
