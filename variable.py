import numpy as np
import math

class Variable:
    """basic variable to store a scalar, perform operations and calculate gradients"""
    def __init__(self, value, _children=()): 
        self.value = value
        self.grad = 0
        self.prev = set(_children)          # children in the reverse computation graph
        self._backward = lambda : None
    
    def __repr__(self):
        """ current state of the object """
        return f"value = {self.value}\ngrad = {self.grad}"
    
    def __str__(self):
        """ Readable repesentation of the object """
        return f"Variable(value={self.value})"
    
   
    def __neg__(self):                      # self * -1
        return self * -1
    
    # fundamental method
    def __pow__(self, power):               # self ^ power
        assert isinstance(power, (int,float))
        out = Variable(self.value**power, (self,))   

        def _backward():
            self.grad += power*(self.value**(power-1)) * out.grad
        
        out._backward = _backward
        return out

    # fundamental method
    def __add__(self, other):               # self + other
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out = Variable(self.value + other.value, (self, other,))   # returns Variable object

        # closure function , remembers the variables from the scope where it was created
        def _backward():
            self.grad += 1.0 *out.grad
            other.grad += 1.0 *out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        return Variable(self.value + other.value)   # returns Variable object
    
    def __sub__(self, other):               # self + (-other)
        return self + (-other)
    
    def __rsub__(self, other):               # other + (-self) 
        return -self + other
    
    # fundamental method
    def __mul__(self, other):               # self * other
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out = Variable(self.value * other.value, (self, other,))   # returns Variable object

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):               # other * self
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out =  Variable(self.value * other.value, (self, other))   # returns Variable object
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def exp(self):
        out = Variable(math.exp(self.value), (self,))

        def _backward():
            self.grad += out.value * out.grad     # y = e^x  dy/dx = e^x (-> out.value)  out.grad -> out's grad
        out._backward = _backward
        return out

    def sigmoid(self):
        # e^x/1+e^x
        t = self.exp().value
        d = 1 + t
        out = Variable(t/d, (self,))

        def _backward():
            self.grad += out.grad * out.value*(1-out.value)
        out._backward = _backward
        return out

    def relu(self):
        out = Variable(max(0, self.value), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def backprop(self):
        # backprop from self to all the variables in computation graph
        # using topological sort

        comp_graph = []
        self.grad = 1
        visited = set()
        def topo(z):
            if z not in visited:
                comp_graph.append(z)
                visited.add(z)
                for child in z.prev:
                    topo(child)
        topo(self)
        
        for node in comp_graph:
            node._backward()


