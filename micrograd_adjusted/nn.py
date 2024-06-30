import os
import math
import random
from micrograd_adjusted.engine import Value
from graphviz import Digraph


class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self._prev = set()
        self.label = None
        self.data = f"w: {self.w}, b: {self.b}"

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
        self.all_layers = [Layer(nin, nin)] + self.layers
        self.nodes = set()
        self.edges = set()
        
        self.connect()
        self.drawn = False

    def connect(self):
        # adding prev nodes and their labels for hidden layers
        nodes, edges = set(), set()
        for i in range(1, len(self.all_layers)):
            layer = self.all_layers[i]
            prev_layer = self.all_layers[i-1]

            for j, neuron in enumerate(layer.neurons):
                neuron.label = f"n{i}{j+1}"
                neuron._prev = set(prev_layer.neurons)
        
        # adding labels for input layer
        input_layer = self.all_layers[0]
        for j, neuron in enumerate(input_layer.neurons):
            neuron.label = f"x{j+1}"

        roots = self.all_layers[-1].neurons

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        for root in roots:
            build(root)

        self.nodes = nodes
        self.edges = edges

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
    
    def draw(self, format='svg'):
        dot = Digraph(format=format, graph_attr={'rankdir': 'LR'})

        dot.attr('node', shape='circle', width='0.5')
        dot.attr('edge')
        
        nodes, edges = self.nodes, self.edges
        for n in nodes:
            uid = str(id(n))
            if n.label[0] == 'n':
                dot.node(name=uid, label="%s" % (n.label), shape='circle', style='filled')
            elif n.label[0] == 'x':
                dot.node(name=uid, label="%s" % (n.label), shape='box')

            # if n.label[0] == 'n':
            #     dot.node(name=uid, label="{%s | %s}" % (n.label, n.data), shape='record')
            # elif n.label[0] == 'x':
            #     dot.node(name=uid, label="{%s}" % (n.label), shape='record')

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)), len='2.0', arrowsize='0.5')

        if not self.drawn:
            directory_path = 'nn-output'
            os.makedirs(directory_path, exist_ok=True)
            self.draw = True

        output_path = dot.render("nn-output/nn-graph-out", view=False)
        print(f"Graph rendered to {output_path}")

        if os.name == 'nt':
            os.startfile(output_path)
        elif os.name == 'posix':
            os.system(f'open {output_path}')

        return dot


