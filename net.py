from weakref import WeakKeyDictionary
from functools import partial


class Synapse:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        left.right_synapses.append(self)
        right.left_synapses.append(self)
        self.weight = .2
    
    def link_value(self, activation):
        return self.right.activation_value(activation) * self.weight
    
    def interrogate(self, activation):
        return partial(self.right.interrogate, activation)
    
    def reinforce(self, activation):
        # decrease weight, propagate right
        pass


class Neuron:

    def __init__(self):
        self.activations = WeakKeyDictionary()
        self.left_synapses = []
        self.right_synapses = []
    
    def activation_value(self, activation):
        return self.activations.get(activation, 0)
    
    def interrogate(self, synapse, activation):
        activation_value = self.activations.get(activation, 0)
        if  activation_value < 1:
            link_value = synapse.link_value(activation)
            activation_value += link_value
            if activation_value > 1:
                activation_value = 1
            self.activations[activation] = activation_value
            return [right.interrogate(activation) for right in self.right_synapses]
        return []
    
    def reinforce(self, synapse, activation):
        # propagate right
        pass


class Input(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def activation_value(self, activation):
        return 1
    
    def interrogate(self, synapse, activation):
        return [right.interrogate(activation) for right in self.right_synapses]
    
    def reinforce(self, synapse, activation):
        return []
    

class Output(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def interrogate(self, synapse, activation):
        activation_value = self.activations.get(activation, 0)
        if  activation_value < 1:
            link_value = synapse.link_value(activation)
            activation_value += link_value
            self.activations[activation] = activation_value
        # when an output is fully activated, activation is complete
        if activation_value >= 1:
            return self
        return []


class Activation:

    def __init__(self, *inputs):
        self.inputs = inputs
        self.output = None
    
    def resolve(self, max_steps=1000):
        next_iteration = [interrogation for input_ in self.inputs for interrogation in input_.interrogate(None, self)]
        if self.output is None:
            _next_iteration = []
            for _ in range(max_steps):
                for interrogate in next_iteration:
                    results = interrogate()
                    for result in results:
                        if isinstance(result, list):
                            _next_iteration.extend(result)
                        else:
                            self.output = result
                    if self.output is not None:
                        break
                if self.output is not None:
                    break
            else:
                raise RuntimeError('unresolvable')
        return self.output