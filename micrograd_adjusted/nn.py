import math
import random
from engine import Value


class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # 
        # return params


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def fit(self, X, y, iters=1000, learning_rate=0.01):
        for k in range(iters):
            # fwd pass
            ypred = [self(x) for x in X]
            loss = sum((ygt - yout)**2 for ygt, yout in zip(y, ypred))

            # resetting grad
            for p in self.parameters():
                p.grad = 0.0

            # back pass
            loss.backward()

            # grad des
            for p in self.parameters():
                p.data += -learning_rate * p.grad

            if k % (iters // 10) == 0:
                print(f"Iteration: {k}, Loss: {loss.data}")
        
    def predict(self, X):
        return [self(x).data for x in X]
