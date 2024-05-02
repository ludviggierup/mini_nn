from math import exp, log


class Value():

    def __init__(self, data, _children=(), op=''):
        self.data = data
        self.grad = 0.0
        self.op = op
        self._prev = set(_children)
        self._backward = lambda: None

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data: {self.data}, grad: {self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting float or int for now"
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "'relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        e2x = exp(2*x)
        out = Value((e2x - 1) / (e2x + 1), (self,), op='tanh')
        def _backward():
            self.grad += 1 - (out.data)**2 * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(exp(self.data), (self,), op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(log(self.data), (self,), op='log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1




# a = Value(2.0)
# b = Value(3.0)
# c = Value(4.0)
# d = (a+b)*c - b*a + 3*a**2 / 4 + a.exp()
# vals = [a,b,c,d]
# d.backward()
# print([v.grad for v in vals])
#

