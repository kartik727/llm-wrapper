from llm_wrappers.wrappers.base_wrapper import ChatLLMWrapper
from llm_wrappers.io_objects.base_io_object import BaseChatObject

def chat_ui(
        model:ChatLLMWrapper, sys_prompt:str, /, *,
        exit_msg:str='EXIT', context:BaseChatObject=None
    )->BaseChatObject:
    """A simple command line interface for chatting with a model.

    Args:
        model (ChatLLMWrapper): The LLM model to chat with.
        sys_prompt (str): The system prompt to use.
        exit_msg (str, optional): The message to use to exit the chat. Defaults
            to 'EXIT'.
        context (BaseChatObject, optional): The context to use for the chat. If
            None, a new (empty) context will be created. Defaults to None.

    Returns:
        BaseChatObject: The context after the chat has ended.
    """
    if context is None:
        context = model.new_chat(sys_prompt)
    while True:
        user_input = input('You: ')
        if user_input == exit_msg:
            break
        context, response = model.chat(context, user_input)
        print(f'\nAssistant: {response}\n')
    return context
