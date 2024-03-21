"""Objects for interacting with LLMs."""

from .constants.modalities import TEXT


class BaseIO:
    """Base class for I/O objects."""

    def __init__(self, items: list[tuple]):
        """Initializes the input object.

        Args:
            items (list[tuple]): The list of items in the input.
        """
        self._items = items

    @property
    def items(self):
        """The list of items in the input."""
        return self._items


class ChatInput(BaseIO):
    """Input object for chat-based LLMs."""

    def __init__(self, prompt: str, *,
                 history: list[tuple] = None, sys_prompt: str = None):
        """Initializes the chat input object.

        Args:
            prompt (str): The user prompt.
            history (list[tuple], optional): The history. Defaults to None.
            sys_prompt (str, optional): The system prompt. Defaults to None.
        """

        items = []
        if sys_prompt is not None:
            items.append((sys_prompt, TEXT, {'role': 'system'}))

        if history is not None:
            for (p, r) in history:
                items.append((p, TEXT, {'role': 'user'}))
                items.append((r, TEXT, {'role': 'assistant'}))

        items.append((prompt, TEXT, {'role': 'user'}))
        super().__init__(items)

    @property
    def prompt(self) -> str:
        return self.items[-1][0]


class ChatOutput(BaseIO):
    """Output object for chat-based LLMs."""

    def __init__(self, input: ChatInput, response: str, *,
                 modality: str = TEXT, role: str = 'assistant', **kwargs):
        """Initializes the chat output object.

        Args:
            input (ChatInput): The input object.
            response (str): The response from the LLM.
        """
        items = input.items + [(response, modality, {'role': role, **kwargs})]
        super().__init__(items)

    @property
    def text(self):
        """The response from the LLM."""
        return self.items[-1][0]


class Input(BaseIO):
    """Common input object for LLMs."""

    @staticmethod
    def chat(prompt: str, *,
             history: list[tuple] = None, sys_prompt: str = None):
        """Creates a chat input object.

        Args:
            prompt (str): The user prompt.
            history (list[tuple], optional): The history. Defaults to None.
            sys_prompt (str, optional): The system prompt. Defaults to None.

        Returns:
            ChatInput: The chat input object.
        """
        return ChatInput(prompt, history=history, sys_prompt=sys_prompt)
