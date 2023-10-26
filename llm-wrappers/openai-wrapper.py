import openai
from dataclasses import dataclass
from enum import Enum
from openai.error import ServiceUnavailableError
import time
import logging
import shlex
import json
import pandas as pd

openai.api_key = '' # Enter you API key here
model_name = 'gpt-3.5-turbo'

logging.basicConfig(level=logging.DEBUG)

class UnknownResponseError(Exception):
    pass

class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    FUNCTION = 'function'

@dataclass
class Message:
    role: Role
    text: str

    @property
    def formatted(self):
        return {'role' : self.role.value, 'content' : self.text}


@dataclass
class FunctionResponse:
    role: Role
    name: str
    text: str

    @property
    def formatted(self):
        return {'role' : self.role.value, 'name' : self.name, 'content' : self.text}

@dataclass
class FunctionCall:
    role: Role
    params: dict

    @property
    def formatted(self):
        return {'role' : self.role.value, 'content' : None, 'function_call' : self.params}

class Assistant:
    SYS_TXT = 'You are a helpful assistant.'

    def __init__(self, model_name:str, *,
            sys_txt:str=None, retry_wait_time:float=5, functions:list=None):
        self.model_name = model_name
        self.sys_txt = self.SYS_TXT if sys_txt is None else sys_txt
        self.current_chat = []
        self.retry_wait_time = retry_wait_time
        self.response_logs = []
        self.reset()
        self._n_prompt_tokens = 0
        self._n_completion_tokens = 0
        self._functions = functions

    def _append_exchange(self, prompt:Message | FunctionResponse, ast_reply:Message | FunctionCall):
        assert prompt.role in [Role.USER, Role.FUNCTION], \
            f'Invalid role for `prompt`: {prompt.role}'
        assert ast_reply.role == Role.ASSISTANT, \
            f'Invalid role for `ast_reply`: {ast_reply.role}'
        self.current_chat.append(prompt.formatted)
        self.current_chat.append(ast_reply.formatted)

    @property
    def chat_kwargs(self)->dict:
        kwargs = dict()

        if self._functions is not None:
            kwargs['functions'] = self._functions

        return kwargs

    def _prompt(self, msg:Message | FunctionResponse)->Message | FunctionCall:
        # Get the response. Keep retrying until success
        success = False
        while not success:
            try:
                ast_result = openai.ChatCompletion.create(
                    model = self.model_name,
                    messages = self.current_chat + [msg.formatted],
                    **self.chat_kwargs)
                self._n_prompt_tokens += ast_result['usage']['prompt_tokens']
                self._n_completion_tokens += ast_result['usage']['completion_tokens']

                success = True
                self.response_logs.append(ast_result)
            except ServiceUnavailableError:
                logging.warning(f'Service not available. Retrying after {self.retry_wait_time} s.')
                time.sleep(self.retry_wait_time)

        # Parse the response
        ast_response = ast_result['choices'][0]
        assert ast_response['message']['role'] == 'assistant', \
            f'Prompt response must be from `assistant`, not `{ast_response["message"]["role"]}`'

        if ast_response['finish_reason'] == 'stop':
            ast_txt = ast_response['message']['content']
            logging.debug(f'Prompt response (text): {ast_txt}')
            return Message(Role.ASSISTANT, ast_txt)
        elif ast_response['finish_reason'] == 'function_call':
            ast_fc_params = ast_response['message']['function_call']
            logging.debug(f'Prompt response (func call): {ast_fc_params}')
            return FunctionCall(Role.ASSISTANT, ast_fc_params)
        else:
            raise UnknownResponseError(f'Messages with `finish_reason`==`{ast_response["finish_reason"]}` cannot be parsed yet.')

    def _function_processor(self, func_call:FunctionCall)->FunctionResponse:
        raise NotImplementedError('`_function_processor` has not been implemented')

    @property
    def n_prompt_token(self):
        return self._n_prompt_tokens

    @property
    def n_completion_tokens(self):
        return self._n_completion_tokens

    @property
    def n_total_tokens(self):
        return self._n_prompt_tokens + self._n_completion_tokens

    def reset(self):
        sys_msg = Message(Role.SYSTEM, self.sys_txt)
        self._n_prompt_tokens = 0
        self._n_completion_tokens = 0
        self.response_logs = []
        self.current_chat = [sys_msg.formatted]

    def chat(self, text_prompt:str)->str:
        usr_prompt = Message(Role.USER, text_prompt)

        ast_reply = self._prompt(usr_prompt)
        self._append_exchange(usr_prompt, ast_reply)

        # assistant calls functions until it has all the info it needs
        while isinstance(ast_reply, FunctionCall):
            func_response = self._function_processor(ast_reply)
            ast_reply = self._prompt(func_response)
            self._append_exchange(func_response, ast_reply)

        # finally it returns a response to the user
        if isinstance(ast_reply, Message):
            return ast_reply.text

        # any other type of response is invalid
        else:
            raise TypeError(f'Unknown prompt reply type: `{type(ast_reply)}`')

        reply_for_user = self._get_reply(ast_reply)
        return reply_for_user

    def chat_fresh(self, text_prompt:str)->str:
        self.reset()
        logging.debug('Chat has been reset.')
        return self.chat(text_prompt)

    def chat_ui(self):
        usr_chat = ''
        self.reset()
        print(f'Welcome to chat. Enter `EXIT` to exit.\n')
        while usr_chat != 'EXIT':
            usr_chat = input('\n')
            print(self.chat(usr_chat))

        print('Chat over.')

class InvalidResponseError(Exception):
    pass

class EndChatInterrupt(Exception):
    pass

class SysDB:
    def __init__(self, order_table:pd.DataFrame, product_table:pd.DataFrame):
        self.order_table = order_table
        self.product_table = product_table

    def get_order_status(self, order_number:int)->str:
        return self.order_table.loc[order_number]['order_status']

    def check_refund_available(self, order_number:int)->bool:
        return self.order_table.loc[order_number]['refund_available']

    def get_product_details(self, product_number:int)->dict:
        return self.product_table.loc[product_number].to_dict()

    def get_recommendations(self, product_type:str)->list[int]:
        products = self.product_table[self.product_table['product_type']==product_type]
        return [x[0] for x in products.iterrows()]

class SysProcessor:
    def __init__(self, sys_db:SysDB):
        self.sys_db = sys_db

    def is_shipping_available(self, city_name:str, state_code:str)->str:
        return str(True)

    def get_order_status(self, order_id:str)->str:
        return self.sys_db.get_order_status(int(order_id))

    def check_refund_available(self, order_id:str)->str:
        return str(self.sys_db.check_refund_available(int(order_id)))

    def get_product_details(self, product_id:str)->str:
        return json.dumps(self.sys_db.get_product_details(int(product_id)))

    def get_product_suggestions(self, category:str)->str:
        return json.dumps(self.sys_db.get_recommendations(category))

    def request_transfer_to_human(self):
        raise EndChatInterrupt('The chat is being transferred to a human expert.')

    def sys_request(self, query:str)->str:
        return input(f'Message by AI for human expert: ```{query}```\n')

    def chat_complete(self, issue_resolved:str, customer_satisfaction_level:str):
        self.response_dict = {'issue_resolved':issue_resolved,
            'customer_satisfaction_level':customer_satisfaction_level}
        raise EndChatInterrupt('Chat has been marked complete by AI.')

@dataclass
class Customer:
    name: str
    account_type: str
    city: str
    account_age_months: int

@dataclass
class Order:
    order_id: int
    product_id: int
    payment_type: str

class CustomerIssue:
    def __init__(self, customer:Customer, *, order:Order=None):
        self.customer = customer
        self.order = order

    def json_str(self)->str:
        cust_data = {
            'name' : self.customer.name,
            'account_type' : self.customer.account_type,
            'city' : self.customer.city,
            'account_age_months' : self.customer.account_age_months,
        }
        if self.order is None:
            order_data = None
        else:
            order_data = {
                'order_id' : self.order.order_id,
                'product_id' : self.order.product_id,
                'payment_type' : self.order.payment_type,
            }
        return json.dumps({'customer': cust_data, 'order': order_data})
    
class CustomerSupportAssistant(Assistant):

    def __init__(self, model_name:str, sys_processor:SysProcessor, *,
            sys_txt:str=None, retry_wait_time:float=5, functions:list=None):
        super().__init__(model_name, sys_txt=sys_txt,
            retry_wait_time=retry_wait_time, functions=functions)
        self.sys_processor = sys_processor
        self._status = 'In progress'

    @property
    def status(self):
        return self._status

    def reset(self):
        super().reset()
        self._status = 'In progress'

    def _function_processor(self, func_call:FunctionCall)->FunctionResponse:
        func_name = func_call.params['name']
        func_kwargs = json.loads(func_call.params['arguments'])
        func_response = getattr(self.sys_processor, func_name)(**func_kwargs)
        return FunctionResponse(Role.FUNCTION, func_name, func_response)

    def chat_ui(self, cust_issue:CustomerIssue):
        self.reset()
        print(f'Welcome to chat. Enter `EXIT` to exit.\n')
        try:
            # send initial data about the customer
            ast_response = self.chat(cust_issue.json_str())

            # begin chat
            while True:
                usr_chat = input('\n\nDarry: ' + ast_response + '\n' + '-'*80 + '\n')
                if usr_chat == 'EXIT':
                    self._status = 'Ended by user'
                    raise EndChatInterrupt('User has ended that chat.')
                ast_response = self.chat(usr_chat)

        except EndChatInterrupt as e:
            print(e)
            print('Thank you for chatting with us.')
        finally:
            print('Chat over.')

sys_prompt = f'''You are a helpful Customer Support AI Assistant called Darry \
the AI Bot, or Darry for short. Your job is to provide support for customers \
of the US-based e-commerce website Bamazon Shopping.

You will first receive a message from the system with a JSON formatted list of name-value pairs containing \
some relevant information about the customer, which you can reference during the \
conversation.

Then, you will start chatting with the customer. You should first greet the \
customer, mention that you are an AI customer support agent, and then ask them how you can \
provide support to them. You should chat with the customer in a brief and \
professional manner with a natural and polite tone.

During the chat, you can make function calls to request additional information. \
If you need some information that is not provided by one of the available functions, \
you can call the `sys_request` function to make a query in natural language that will be \
processed by a human operator.

Finally, after resolving the customer issue, thank them for their patience and apologize for any \
inconvenience that was caused to them. Once they have responded to the thank you message, \
you must call the `chat_complete` function with an analysis of the chat, to let the system know
that that chat is over.
'''

functions = [
    {
    'name': 'is_shipping_available',
    'description': 'Whether items can be shipped to city `city_name` in state `state_code`. Returns "True" or "False"',
    'parameters': {
        'type': 'object',
        'properties': {
            'city_name': {
                'type': 'string',
                'description': 'The city, e.g. San Francisco',
            },
            'state_code': {
                'type': 'string',
                'description': 'The two-character state code, e.g. CA'
            },
        },
        'required': ['city_name', 'state_code'],
        }
    },
    {
    'name': 'get_order_status',
    'description': 'Get status of the order with the given `order_id`. Returns the status as a string.',
    'parameters': {
        'type': 'object',
        'properties': {
            'order_id': {
                'type': 'string',
                'description': 'Order ID of the order. E.g. 100433',
            }
        },
        'required': ['order_id'],
        }
    },
    {
    'name': 'check_refund_available',
    'description': 'Check if the order with the given `order_id` is eligible for a refund. Returns "True" or "False"',
    'parameters': {
        'type': 'object',
        'properties': {
            'order_id': {
                'type': 'string',
                'description': 'Order ID of the order. E.g. 100433',
            }
        },
        'required': ['order_id'],
        }
    },
    {
    'name': 'get_product_details',
    'description': 'Get details of the product with the given `product_id` in a JSON formatted string',
    'parameters': {
        'type': 'object',
        'properties': {
            'product_id': {
                'type': 'string',
                'description': 'Product ID of the product. E.g. 503423',
            }
        },
        'required': ['product_id'],
        }
    },
    {
    'name': 'get_product_suggestions',
    'description': 'Get suggestions for products for the given `category`. Returns a string containing a comma-separated list of product ids.',
    'parameters': {
        'type': 'object',
        'properties': {
            'cateogory': {
                'type': 'string',
                'description': 'Category of products, e.g. Toy. Category of a product can be retrieved using the `get_product_details` function.',
            }
        },
        'required': ['category'],
        }
    },
    {
    'name': 'sys_request',
    'description': 'Make a query in natural language to a human operator, for information that is not available by other functions',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Query in natural language for a human operator.',
            }
        },
        'required': ['query'],
        }
    },
    {
    'name': 'request_transfer_to_human',
    'description': 'End the current chat and transfer to human support expert',
    'parameters': {
        'type': 'object',
        'properties': {},
        }
    },
    {
    'name': 'chat_complete',
    'description': 'Call this function with the analysis of the chat when the chat with the customer is over.',
    'parameters': {
        'type': 'object',
        'properties': {
            'issue_resolved': {
                'type': 'string',
                'description': 'Whether the customer issue was resolved. Must be "True" or "False"',
            },
            'customer_satisfaction_level': {
                'type': 'string',
                'description': 'The satisfaction level of the customer with the chat support. Must be "Positive", "Neutral" or "Negative"',
            }
        },
        'required': ['issue_resolved', 'customer_satisfaction_level'],
        }
    },
]

order_table = pd.DataFrame({
    'order_status' : ['Delivered', 'Order Recieved', 'In Transit', 'Delivered', 'In Transit - Delayed', 'Return Initiated', 'Return Complete'],
    'refund_available' : [False, True, True, True, True, True, False]
    },
    index = [10045, 10046, 10047, 10048, 10049, 10050, 10051])

product_table = pd.DataFrame({
    'product_type' : ['Toy', 'TV', 'TV', 'Toy', 'TV', 'Garden Equipment'],
    'product_name' : ['MyFluffy Soft Panda', 'Fefenex 32" HD TV', 'Tamtung LED TV Ultra 55" 4K', 'SuperBoys Soldier | toy with lights and sound', 'Jempis Hatak HiQuality 31.5" TV', 'Reds Garden Glove'],
    'product_price' : [12.99, 130.00, 439.99, 8.49, 119.99, 5.99]   # Should be decimals, not floats. But we don't care in this dummy implementation.
    },
    index=[51033, 51034, 51035, 51036, 51037, 51038])

cust = Customer('Sara B', 'Individual', 'Miami, FL', 2)
order = Order(10050, 51034, 'Credit Card')
c_issue = CustomerIssue(cust, order=order)

sys_db = SysDB(order_table, product_table)
sp = SysProcessor(sys_db)
bot = CustomerSupportAssistant(model_name, sp, sys_txt=sys_prompt, functions=functions)

bot.chat_ui(c_issue)