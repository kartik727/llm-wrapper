"""The LLM Wrappers package.

This package contains wrappers with a unified interface for various
Large Language Models (LLMs).
"""

import logging

try:
    from llm_wrappers.wrappers.openai_wrapper import OpenAIWrapper
except ImportError:
    logging.warning('OpenAIWrapper not imported. OpenAI API not available.')

try:
    from llm_wrappers.wrappers.hf_wrapper import (
        HFWrapper, LlamaWrapper, ZephyrWrapper, PythiaWrapper)
except ImportError:
    logging.warning('LlamaWrapper not imported. Llama API not available.')
