import torch as t
from collections import defaultdict

class Ops:

    def __add__(self, other):
        return Add(self, other)
    
    def __mul__(self, other):
        return Mul(self, other)
    
    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Div(self, other)


class Var(Ops):
    
    def __init__(self, value):
        self.value = value

class Add(Ops):

    def __init__(self, a:Var, b:Var):
        self.value = a.value + b.value
        self.grad = [(a,1), (b, 1)]
    
class Mul(Ops):

    def __init__(self, a:Var, b:Var):
        self.value = a.value * b.value
        self.grad = [(a, b.value), (b, a.value)]

class Sub(Ops):

    def __init__(self, a:Var, b:Var):
        self.value = a.value - b.value
        self.grad = [(a, 1), (-b, 1)]

class Div(Ops):

    def __init__(self, a:Var, b:Var):
        self.value = a.value/b.value
        self.grad = [()]

def gradients_val(parent_node):

    gradients = defaultdict(lambda : 0)
    stack = parent_node.grad.copy()

    while stack:
        node, value = stack.pop()
        gradients[node] += value

        if not isinstance(node, Var):
            for child_node, child_route_value in node.grad:
                stack.append((child_node, child_route_value * value))

    return dict(gradients)  