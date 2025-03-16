from weakref import WeakKeyDictionary
from functools import partial
from collections import deque, namedtuple
import random
from itertools import chain, permutations, cycle


class Synapse:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        left.right_synapses.append(self)
        right.left_synapses.append(self)
        self.weight = .2
        self.activations = WeakKeyDictionary()
    
    def marginal_link_value(self, activation):
        total_value = self.left.activation_value(activation) * self.weight
        previous_value = self.activations.get(activation, 0)
        self.activations[activation] = total_value
        return total_value - previous_value
    
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
        marginal_link_value = inbound_synapse.marginal_link_value(activation)
        if  marginal_link_value > .000001:
            activation_value = self.activations.get(activation, 0) + marginal_link_value
            if activation_value > 1:
                self.activations[activation] = 1
            else:
                self.activations[activation] = activation_value
            return ([outbound_synapse.interrogate(activation) for outbound_synapse in self.right_synapses], None)
        return ([], None)


class Input(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def activation_value(self, activation):
        return 1
    
    def interrogate(self, inbound_synapse, activation):
        return ([right.interrogate(activation) for right in self.right_synapses], None)
    

class Output(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def interrogate(self, inbound_synapse, activation):
        marginal_link_value = inbound_synapse.marginal_link_value(activation)
        if  marginal_link_value > .000001:
            activation_value = self.activations.get(activation, 0) + marginal_link_value
            if activation_value > 1:
                self.activations[activation] = 1
            else:
                self.activations[activation] = activation_value
        return ([], self)


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
        if self.synapse is not None:
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


ScoredOutput = namedtuple('ScoredOutput', ['output', 'score'])
IOData = namedtuple('IOData', ['inputs', 'scored_outputs'])


class Activation:

    def __init__(self, *inputs):
        self.inputs = inputs
        self.outputs = None
    
    def resolve(self, max_steps=1000, print_sampling_frequency=100):
        current_iteration = [interrogation for input_ in self.inputs for interrogation in input_.interrogate(None, self)[0]]
        if self.outputs is None:
            outputs = set()
            for i in range(max_steps):
                next_iteration = []
                if i % print_sampling_frequency == 0:
                    print(i, len(current_iteration))
                for interrogate in current_iteration:
                    results, output = interrogate()
                    next_iteration.extend(results)
                    if output is not None:
                        outputs.add(output)
                if not next_iteration:
                    break
                current_iteration = next_iteration
            self.outputs = [ScoredOutput(o, o.activation_value(self)) for o in outputs]
            self.outputs.sort(key=lambda so: so.score, reverse=True)
        return self.outputs
    
    def reinforce(self, factor, target_output, max_steps=10000):
        if self.outputs is None:
            raise RuntimeError('solution has not been resolved')
        for scored_output in self.outputs:
            if scored_output.output is target_output:
                break
        else:
            raise ValueError('target_output provided is not in the output group for this activation')
        probes = deque()
        probes.append(ReinforceProbe(target_output, self))
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
    
    def io_data(self):
        # convenience method to provide an object representing the io of the Activation that allows
        # references to the activation to be released, so solutions can be stored but still allow
        # for the activation dictionaries to be flushed
        return IOData(self.inputs, self.outputs)


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


# network = Network(('1',), ('1',), 2, 3, lambda *_: 1)
# activation = network.spawn_activation('1')
# print(activation.resolve(max_steps=10))
# print(network.outputs['1'].activation_value(activation))


network = Network(('0', '1', '2', '+'), ('1', '2', '3'), 21, 126, lambda *_: 7)
scenarios = iter(cycle(((('0', '1', '+'), '1'),(('0', '2', '+'), '2'),(('2', '1', '+'), '3'))))
for i in range(30):
    print(f'------ {i} ------')
    inputs, expected = next(scenarios)
    activation = network.spawn_activation(*inputs)
    result = activation.resolve(max_steps=100, print_sampling_frequency=10)
    for output in result:
        print(output.score, output.output.value)
        if output.output.value == expected:
            print('^correct^')
            activation.reinforce(.5, output.output)
        else:
            activation.reinforce(-.5, output.output)


def test_neuron_activation_value_0():
    neuron = Neuron()
    activation = Activation()
    neuron.activations[activation] = 0
    assert neuron.activation_value(activation) == 0

def test_neuron_activation_value_decimal():
    neuron = Neuron()
    activation = Activation()
    neuron.activations[activation] = .5
    assert neuron.activation_value(activation) == .5

def test_neuron_activation_value_1():
    neuron = Neuron()
    activation = Activation()
    neuron.activations[activation] = 1
    assert neuron.activation_value(activation) == 1

def test_neuron_activation_value_unset():
    neuron = Neuron()
    activation = Activation()
    assert neuron.activation_value(activation) == 0

def test_weakref_release_for_activation_value():
    import gc
    neuron = Neuron()
    activation = Activation()
    neuron.activations[activation] = .5
    assert neuron.activation_value(activation) == .5
    del activation
    gc.collect()
    assert len(neuron.activations) == 0

def test_synapse_initialization():
    neuron_1 = Neuron()
    neuron_2 = Neuron()
    synapse_1 = Synapse(neuron_1, neuron_2)
    assert synapse_1.left is neuron_1
    assert synapse_1.right is neuron_2
    assert neuron_1.left_synapses == []
    assert neuron_1.right_synapses == [synapse_1]
    assert neuron_2.left_synapses == [synapse_1]
    assert neuron_2.right_synapses == []
    neuron_3 = Neuron()
    synapse_2 = Synapse(neuron_1, neuron_3)
    synapse_3 = Synapse(neuron_3, neuron_2)
    assert synapse_1.left is neuron_1
    assert synapse_1.right is neuron_2
    assert synapse_2.left is neuron_1
    assert synapse_2.right is neuron_3
    assert synapse_3.left is neuron_3
    assert synapse_3.right is neuron_2
    assert neuron_1.left_synapses == []
    assert neuron_1.right_synapses == [synapse_1, synapse_2]
    assert neuron_2.left_synapses == [synapse_1, synapse_3]
    assert neuron_2.right_synapses == []
    assert neuron_3.left_synapses == [synapse_2]
    assert neuron_3.right_synapses == [synapse_3]
    
def test_synapse_link_value():
    neuron_1 = Neuron()
    neuron_2 = Neuron()
    synapse = Synapse(neuron_1, neuron_2)
    synapse.weight = .5
    activation = Activation()
    neuron_1.activations[activation] = .5
    neuron_2.activations[activation] = .2
    assert synapse.marginal_link_value(activation) == .25

def test_neuron_interrogation_not_yet_activated():
    inbound_neuron = Neuron()
    neuron = Neuron()
    outbound_neuron = Neuron()
    inbound_synapse = Synapse(inbound_neuron, neuron)
    inbound_synapse.weight = .5
    outbound_synapse = Synapse(neuron, outbound_neuron)
    activation = Activation()
    inbound_neuron.activations[activation] = .5
    outbound_neuron.activations[activation] = .25
    return_value = neuron.interrogate(inbound_synapse, activation)
    assert len(return_value) == 1
    assert return_value[0].func.__self__ is outbound_neuron
    assert neuron.activation_value(activation) == .25

def test_neuron_interrogation_already_activated():
    inbound_neuron = Neuron()
    neuron = Neuron()
    outbound_neuron = Neuron()
    inbound_synapse = Synapse(inbound_neuron, neuron)
    inbound_synapse.weight = .5
    outbound_synapse = Synapse(neuron, outbound_neuron)
    activation = Activation()
    inbound_synapse.activations[activation] = .1
    inbound_neuron.activations[activation] = .5
    neuron.activations[activation] = .1
    outbound_synapse.activations[activation] = .25
    outbound_neuron.activations[activation] = .35
    return_value = neuron.interrogate(inbound_synapse, activation)
    assert len(return_value) == 1
    assert return_value[0].func.__self__ is outbound_neuron
    assert neuron.activation_value(activation) == .25

def test_neuron_interrogation_already_fully_activated():
    inbound_neuron = Neuron()
    neuron = Neuron()
    outbound_neuron = Neuron()
    inbound_synapse = Synapse(inbound_neuron, neuron)
    inbound_synapse.weight = .5
    outbound_synapse = Synapse(neuron, outbound_neuron)
    activation = Activation()
    inbound_synapse.activations[activation] = .1
    inbound_neuron.activations[activation] = .5
    neuron.activations[activation] = 1
    outbound_synapse.activations[activation] = .25
    outbound_neuron.activations[activation] = .35
    return_value = neuron.interrogate(inbound_synapse, activation)
    assert len(return_value) == 0
    assert neuron.activation_value(activation) == 1