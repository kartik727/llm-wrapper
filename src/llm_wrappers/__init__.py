import logging

try:
    from llm_wrappers.wrappers.openai_wrapper import OpenAIWrapper
except ImportError:
    logging.warning('OpenAIWrapper not imported. OpenAI API not available.')

try:
    from llm_wrappers.wrappers.llama_wrapper import LlamaWrapper
except ImportError:
    logging.warning('LlamaWrapper not imported. Llama API not available.')
