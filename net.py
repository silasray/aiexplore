from weakref import WeakKeyDictionary
from functools import partial
from collections import deque


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
    
    def reinforce(self, activation, factor):
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
    
    def reinforce(self, synapse, activation, factor):
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
    
    def reinforce(self, synapse, activation, factor):
        pass


class ReinforceProbe:

    def __init__(self, neuron, activation, parent=None):
        self.neuron = neuron
        self.activation = activation
        self.parent = parent
        parent_cost = parent.cost if parent else 0
        self.cost = parent_cost + 1 - neuron.activation_value(activation)
    
    def propegate(self):
        for synapse in self.neuron.left_synapses:
            yield  ReinforceProbe(synapse.left, self.activation, self)


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
                next_iteration = _next_iteration
            else:
                raise RuntimeError('unresolvable')
        return self.output
    
    def reinforce(self, factor, max_steps=10000):
        if self.output is None:
            raise RuntimeError('solution has not been resolved')
        probes = deque()
        probes.append(ReinforceProbe(self.output, self))
        paths = {input: [] for input in self.inputs}
        for _ in range(max_steps):
            if not probes:
                break
            probe = probes.pop()
            for new_probe in probe.propegate():
                index = 0
                for index, existing_probe in enumerate(probes):
                    if existing_probe.cost > new_probe.cost:
                        break
                probes.insert(index, new_probe)
            if probe.neuron in paths:
                paths[probe.neuron].append(probe)
        for _, input_probes in paths.items():
