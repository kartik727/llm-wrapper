import json
import pandas as pd

from llm_wrappers import OpenAIWrapper
from llm_wrappers.llm_config import OpenAIConfig
from llm_wrappers.io_objects.openai_io_object import (
    OpenAIFunctionCall, OpenAIFunctionResponse, OpenAIChatObject, Role)

from customer_issue import CustomerIssue

class InvalidResponseError(Exception):
    pass

class EndChatInterrupt(Exception):
    pass

class CustomerServiceBot(OpenAIWrapper):
    def __init__(self, config:OpenAIConfig, /, *, functions:list[dict]=None, 
            database:dict[str, pd.DataFrame]=None):
        super().__init__(config, functions=functions)
        self.database = database
        self.response_dict = None

    def _handle_function_call(self, fc: OpenAIFunctionCall):
        func = getattr(self, fc.name)
        params = json.loads(fc.params)
        func_response = func(**params)
        return OpenAIFunctionResponse(Role.FUNCTION, fc.name, fc.tool_call_id, func_response)
    
    def is_shipping_available(self, city_name:str, state_code:str)->str:
        return str(True)

    def get_order_status(self, order_id:str)->str:
        return str(
            self.database['order_table'].loc[int(order_id)]['order_status'])

    def check_refund_available(self, order_id:str)->str:
        return str(
            self.database['order_table'].loc[int(order_id)]['refund_available'])

    def get_product_details(self, product_id:str)->str:
        return json.dumps(
            self.database['product_table'].loc[int(product_id)].to_dict())

    def get_product_suggestions(self, category:str)->str:
        products = self.database['product_table'][
            self.database['product_table']['product_type']==category]
        return json.dumps([x[0] for x in products.iterrows()])

    def request_transfer_to_human(self):
        raise EndChatInterrupt('The chat is being transferred to a human expert.')

    def sys_request(self, query:str)->str:
        return input(f'Message by AI for human expert: ```{query}```\n')

    def chat_complete(self, issue_resolved:str, customer_satisfaction_level:str):
        self.response_dict = {'issue_resolved':issue_resolved,
            'customer_satisfaction_level':customer_satisfaction_level}
        raise EndChatInterrupt('Chat has been marked complete by AI.') 

    def new_chat(self, 
            sys_prompt:str, customer_issue:CustomerIssue
        )->tuple[OpenAIChatObject, str]:
        context = super().new_chat(sys_prompt)
        context, response = self.chat(context, customer_issue.json_str())
        return context, response
