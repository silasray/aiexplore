from weakref import WeakKeyDictionary
from functools import partial
from collections import deque
import random
from itertools import chain, permutations, cycle


class Synapse:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        left.right_synapses.append(self)
        right.left_synapses.append(self)
        self.weight = .2
    
    def link_value(self, activation):
        return self.left.activation_value(activation) * self.weight
    
    def interrogate(self, activation):
        return partial(self.right.interrogate, self, activation)


class Neuron:

    def __init__(self):
        self.activations = WeakKeyDictionary()
        self.left_synapses = []
        self.right_synapses = []
    
    def activation_value(self, activation):
        return self.activations.get(activation, 0)
    
    def interrogate(self, inbound_synapse, activation):
        activation_value = self.activations.get(activation, 0)
        if  activation_value < 1:
            link_value = inbound_synapse.link_value(activation)
            activation_value += link_value
            if activation_value > 1:
                activation_value = 1
            self.activations[activation] = activation_value
            return [outbound_synapse.interrogate(activation) for outbound_synapse in self.right_synapses]
        return []


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


class ReinforceProbe:

    def __init__(self, neuron, activation, synapse=None, parent=None):
        self.neuron = neuron
        self.synapse = synapse
        self.activation = activation
        self.parent = parent
        parent_cost = parent.cost if parent else 0
        self.cost = parent_cost + 1 - neuron.activation_value(activation)
        self.vistation_weight = synapse.weight if synapse is not None else 0
    
    def propegate(self):
        for synapse in self.neuron.left_synapses:
            yield  ReinforceProbe(synapse.left, self.activation, synapse, self)
    
    def reinforce(self, factor):
        if factor < 0:
            weight = self.synapse.weight + factor * self.vistation_weight
        else:
            weight = self.synapse.weight + factor * (1 - self.vistation_weight)
        if weight < .001:
            self.synapse.weight = .001
        elif weight > 1:
            self.synapse.weight = 1
        else:
            self.synapse.weight = weight


class Activation:

    def __init__(self, *inputs):
        self.inputs = inputs
        self.output = None
    
    def resolve(self, max_steps=1000):
        current_iteration = [interrogation for input_ in self.inputs for interrogation in input_.interrogate(None, self)]
        if self.output is None:
            for _ in range(max_steps):
                next_iteration = []
                for interrogate in current_iteration:
                    results = interrogate()
                    if isinstance(results, list):
                        next_iteration.extend(results)
                    else:
                        self.output = results
                        break
                if not next_iteration and not self.output:
                    raise RuntimeError('unresolvable')
                if self.output is not None:
                    break
                current_iteration = next_iteration
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
        for input_probes in paths.values():
            total_costs = sum(p.cost for p in input_probes)
            for input_probe in input_probes:
                path_factor = factor * (1 - input_probe.cost / total_costs)
                cursor = input_probe
                while cursor.parent is not None:
                    cursor = cursor.parent
                    cursor.reinforce(path_factor)


class Cluster:

    def __init__(self, root):
        self.root = root
        self.neurons = [root]
        self.dangling_neurons = deque(self.neurons)
    
    def generate(self, neuron_count, synapse_count, max_dangling_neurons=5):
        for _ in range(neuron_count):
            if len(self.dangling_neurons) > max_dangling_neurons:
                existing_neuron = self.dangling_neurons.pop()
            else:
                existing_neuron = random.choice(self.neurons)
                if existing_neuron in self.dangling_neurons:
                    self.dangling_neurons.remove(existing_neuron)
            new_neuron = Neuron()
            self.neurons.append(new_neuron)
            self.dangling_neurons.append(new_neuron)
            self.link(existing_neuron, new_neuron)
        for _ in range(synapse_count - neuron_count):
            neuron_1 = random.choice(self.neurons)
            if neuron_1 in self.dangling_neurons:
                self.dangling_neurons.remove(neuron_1)
            neuron_2 = random.choice(self.neurons)
            if neuron_2 in self.dangling_neurons:
                self.dangling_neurons.remove(neuron_2)
            self.link(neuron_1, neuron_2)

    def link(self, neuron_1, neuron_2):
        raise NotImplemented('can only link within input or output cluster')
    
    def get_neuron(self):
        if self.dangling_neurons:
            return self.dangling_neurons.pop()
        return random.choice(self.neurons)


class InputCluster(Cluster):

    def link(self, neuron_1, neuron_2):
        Synapse(neuron_1, neuron_2)


class OutputCluster(Cluster):

    def link(self, neuron_1, neuron_2):
        Synapse(neuron_2, neuron_1)


class Network:

    def __init__(self, inputs, outputs, neuron_count, synapse_count, interlink_factor=lambda i, o, nc, sc: 2):
        self.inputs = {value: Input(value) for value in inputs}
        self.outputs = {value: Output(value) for value in outputs}
        io_count = len(inputs) + len(outputs)
        cluster_size = neuron_count // io_count
        interlink_factor = interlink_factor(inputs, outputs, neuron_count, synapse_count)
        synapse_share = synapse_count // (io_count + interlink_factor)
        clusters = []
        for cluster in chain((InputCluster(i) for i in self.inputs.values()), (OutputCluster(o) for o in self.outputs.values())):
            clusters.append(cluster)
            cluster.generate(cluster_size, synapse_share)
        permutation_iterator = cycle(permutations(clusters, 2))
        for _ in range(synapse_share * interlink_factor):
            left, right = next(permutation_iterator)
            left.link(left.get_neuron(), right.get_neuron())
    
    def spawn_activation(self, *input_values):
        return Activation(*[self.inputs[v] for v in input_values])


network = Network(('1',), ('1',), 2, 3, lambda *_: 1)
activation = network.spawn_activation('1')
for _ in range(1000):
    try:
        print(activation.resolve(max_steps=1000).value)
    except RuntimeError:
        pass
    else:
        break
print(network.outputs['1'].activation_value(activation))


# network = Network(('0', '1', '2', '+'), ('1', '2', '3'), 45, 135)
# activation = network.spawn_activation('0', '1', '+')
# print(activation.resolve(max_steps=100).value)