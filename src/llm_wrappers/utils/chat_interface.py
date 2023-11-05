from llm_wrappers.wrappers.base_wrapper import ChatLLMWrapper
from llm_wrappers.io_objects.base_io_object import BaseChatObject

def chat_ui(
        model:ChatLLMWrapper, sys_prompt:str, exit_msg:str='EXIT'
    )->BaseChatObject:
    context = model.new_chat(sys_prompt)
    while True:
        user_input = input('You: ')
        if user_input == exit_msg:
            break
        context, response = model.chat(context, user_input)
        print(f'Assistant: {response}\n\n')
    return context
