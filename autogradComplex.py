import torch as t

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
    