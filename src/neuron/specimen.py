from dataclasses import dataclass

import torch
from torch import Tensor, nn

from neuron.data_structure import NEURON_DATA_DIM, Data


@dataclass
class Genome:
    # initial latent state for the first neuron(s)
    pluripotent_latent_state: Tensor
    # takes in internal state, outputs activation parameters and health parameter changes
    derive_parameters_from_state: nn.Module
    # takes in internal state, outputs new latent state and state updates
    passive_transform: nn.Module
    # takes in internal state, outputs new latent state and state updates
    activation_transform: nn.Module
    # values in [0, 1] that are continuously multiplied against hormone_influence
    hormone_decay: Tensor
    # takes in internal state pair and relative position (emitter and receiver and emitter to receiver),
    # outputs receptivity of latter [0,1]
    connectivity_coefficient: nn.Module
    # takes in internal state, outputs new latent state and daughter latent state and mitosis direction
    mitosis_results: nn.Module
    # sigmoid(mitosis_health_penalty) / 2 + 0.5 is the cell damage increment upon mitosis
    mitosis_damage: float


class Specimen:
    def __init__(self, genome: Genome):
        self.genome = genome

        initial_neuron_buffer_size = 16
        self.living_neuron_indices = []
        self.dead_neurons_indices = list(range(initial_neuron_buffer_size))
        self.neurons = torch.zeros((initial_neuron_buffer_size, NEURON_DATA_DIM))

    def step(self):
        indices = torch.tensor(self.living_neuron_indices, dtype=torch.int)
        previous_neurons = self.neurons[indices]
        updated_neurons = previous_neurons.clone()

        # handle firing and passive neuron state updates
        self._step_activations(previous_neurons, updated_neurons, indices)

        # decay hormones
        updated_neurons[:, Data.HORMONE_INFLUENCE.value].mul_(self.genome.hormone_decay)

        # absorb hormones
        # handle cell death
        # handle cell division
        # update connectivity
        # update direct parameters
        # update derived parameters

    def _step_activations(self, previous_neurons: Tensor, updated_neurons: Tensor, indices):
        # locate neurons that are ready to fire signals - ion threshold reached and firing warmup completed
        activation_threshold_reached = (
            previous_neurons[:, Data.ACTIVATION_PROGRESS.value]
            >= previous_neurons[:, Data.ACTIVATION_THRESHOLD.value]
        )
        activation_ready = updated_neurons[:, Data.ACTIVATION_WARMUP.value] >= 1
        activated = activation_threshold_reached & activation_ready

        # process the current state of firing neurons and non-firing neurons to get state changes
        activated_state_update = self.genome.activation_transform(previous_neurons[activated, Data.STATE.value])
        passive_state_update = self.genome.passive_transform(previous_neurons[~activated, Data.STATE.value])
        torch.tanh_(activated_state_update[:, Data.TRANSFORM_INCREMENTED.value])
        torch.tanh_(passive_state_update[:, Data.TRANSFORM_INCREMENTED.value])

        # set activation warmup to 0 for neurons that have just fired
        updated_neurons[activated, Data.ACTIVATION_WARMUP.value].zero_()
        # increment activation_warmup, cell_damage, and mitosis_stage
        updated_neurons[activated, Data.INCREMENTED_PARAMETERS.value].add_(
            activated_state_update[:, Data.TRANSFORM_INCREMENTED.value])
        updated_neurons[~activated, Data.INCREMENTED_PARAMETERS.value].add_(
            passive_state_update[:, Data.TRANSFORM_INCREMENTED.value])
        # set the updates to the latent state
        updated_neurons[activated, Data.LATENT_STATE.value].copy_(
            activated_state_update[:, Data.TRANSFORM_LATENT.value])
        updated_neurons[~activated, Data.LATENT_STATE.value].copy_(passive_state_update[:, Data.TRANSFORM_LATENT.value])

        # send signals to destination neurons
        signal_destinations = previous_neurons[activated, Data.OUTPUT_INDICES.value].reshape(-1)
        signal_connectivity = previous_neurons[activated, Data.OUTPUT_CONNECTIVITY.value]
        signal_strengths = (signal_connectivity * previous_neurons[activated, Data.SIGNAL_STRENGTH.value]).reshape(-1)

        cumulative_signals = torch.zeros_like(self.neurons[:, Data.ACTIVATION_PROGRESS.value])
        cumulative_signals.scatter_add_(0, signal_destinations, signal_strengths)
        updated_neurons[:, Data.ACTIVATION_PROGRESS.value].add_(cumulative_signals[indices])

    def add_neurons(self, positions: Tensor, latent_states: Tensor, set_parameters: bool = True):
        """


        Cell damage, mitosis stage, total receptivity, total emissivity, activation progress,
        and hormone influence are initialized as zeros.

        :param positions:
        :param latent_states:
        :param set_parameters:
        :return:
        """
        neuron_indices = self._allocate_neurons(positions.size(0))

        self.neurons[neuron_indices, :] = 0
        self.neurons[neuron_indices, Data.POSITION.value] = positions
        self.neurons[neuron_indices, Data.LATENT_STATE.value] = latent_states

        if set_parameters:
            parameters = self.genome.derive_parameters_from_state(self.neurons[neuron_indices, Data.STATE.value])
            self.neurons[neuron_indices, Data.DERIVED_PARAMETERS.value] = parameters

    def _allocate_neurons(self, neuron_count: int) -> Tensor:
        # allocate more space if necessary
        if len(self.dead_neurons_indices) < neuron_count:
            current_neuron_buffer_size = self.neurons.size(0)

            # keep doubling the total size of the buffer until we have enough space
            extension_size = current_neuron_buffer_size
            extension_factor = 2
            while len(self.dead_neurons_indices) + extension_size < neuron_count:
                extension_size += extension_factor * current_neuron_buffer_size
                extension_factor *= 2
            self.neurons = torch.cat((self.neurons, torch.zeros((extension_size, NEURON_DATA_DIM))))
            self.dead_neurons_indices.extend(
                range(current_neuron_buffer_size, current_neuron_buffer_size + extension_size))
        allocation = self.dead_neurons_indices[:neuron_count]
        self.dead_neurons_indices = self.dead_neurons_indices[neuron_count:]
        self.living_neuron_indices.extend(allocation)
        return torch.tensor(allocation, dtype=torch.int)

    def _deallocate_neurons(self, indices: Tensor):
        indices = set(indices)
        self.living_neuron_indices = list(filter(lambda i: i not in indices, self.living_neuron_indices))
        self.dead_neurons_indices.extend(indices)
