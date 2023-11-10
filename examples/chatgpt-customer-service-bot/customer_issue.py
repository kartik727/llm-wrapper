import json
from dataclasses import dataclass

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