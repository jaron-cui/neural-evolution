from dataclasses import dataclass
from typing import Tuple, List, Iterable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from neuron.data_structure import NEURON_DATA_DIM, Data, NEURON_DIAMETER
from neuron.genome import Genome, InitialNeuronConfiguration


class Specimen:
    def __init__(
        self,
        genome: Genome,
        start_configuration: Iterable[InitialNeuronConfiguration],
        neuron_repulsion: float = 0.01,
        device: torch.device = 'cuda'
    ):
        self.genome = genome.to(device)
        self.neuron_repulsion = neuron_repulsion
        self.device = device
        self.log = StepLog()

        initial_neuron_buffer_size = 16
        self.living_neuron_indices = []
        self.dead_neurons_indices = list(range(initial_neuron_buffer_size))
        self.io_index_map = {}
        self.io_labels = torch.zeros(initial_neuron_buffer_size, device=device, dtype=torch.uint8)
        self.io_label_to_int = {variant: i + 1 for i, variant in enumerate(genome.differentiated_latent_states.keys())}
        self.neurons = torch.zeros((initial_neuron_buffer_size, NEURON_DATA_DIM), device=device)
        self._init_neurons(start_configuration)

    def _init_neurons(self, start_configuration: Iterable[InitialNeuronConfiguration]):
        positions, latents = [], []
        variants = []
        for config in start_configuration:
            if config.differentiation is None:
                latent = self.genome.pluripotent_latent_state
            else:
                latent = self.genome.differentiated_latent_states[config.differentiation]

            positions.append(config.start_position.clone())
            latents.append(latent.clone())
            variants.append(config.differentiation)
        allocated_indices = self.add_neurons(torch.stack(positions), torch.stack(latents), variants=variants)
        self.io_index_map = {i: index.item() for i, index in enumerate(allocated_indices)}

    def step(self):
        self.log = StepLog()
        indices = torch.tensor(self.living_neuron_indices, dtype=torch.int, device=self.device)
        previous_neurons = self.neurons[indices]
        updated_neurons = previous_neurons.clone()

        positions = previous_neurons[:, Data.POSITION.value]
        diffs = positions[:, None, :] - positions[None, :, :]  # shape: (n, n, 3)
        distances = torch.norm(diffs, dim=-1)  # shape: (n, n)
        # print(diffs.shape, distances.shape)
        directions = diffs / distances.unsqueeze(2)
        directions[directions.isnan()] = 0
        connectivity = self._compute_connectivity(previous_neurons, updated_neurons, distances)
        # update neuron positions
        self._handle_physics(updated_neurons, directions, distances, connectivity)
        # handle hormone emission, absorption, and decay
        self._handle_hormones(previous_neurons, updated_neurons, distances)
        # handle firing and passive neuron state updates
        self._handle_state_transform_updates(previous_neurons, updated_neurons, connectivity, indices)

        self.log.neuron_positions = previous_neurons[:, Data.POSITION.value]
        self.log.connectivity = connectivity

        # handle cell death and division
        updated_neurons, indices = self._handle_life_and_death(updated_neurons, indices)

        # update derived parameters
        derived_parameters = self.genome.derive_parameters_from_state(updated_neurons[:, Data.STATE.value])
        updated_neurons[:, Data.DERIVED_PARAMETERS.value] = derived_parameters
        # if we don't normalize hormone emission, we get a positive feedback loop...
        updated_neurons[:, Data.HORMONE_EMISSION.value] = F.normalize(updated_neurons[:, Data.HORMONE_EMISSION.value], dim=1)

        self.neurons[indices] = updated_neurons

        self.log.neuron_count = len(self.living_neuron_indices)

    def _handle_state_transform_updates(
        self, previous_neurons: Tensor, updated_neurons: Tensor, connectivity: Tensor, indices: Tensor
    ):
        # decay ions
        updated_neurons[:, Data.ACTIVATION_PROGRESS.value] *= self.genome.activation_decay

        # locate neurons that are ready to fire signals - ion threshold reached and firing warmup completed
        activation_threshold_reached = (
            previous_neurons[:, Data.ACTIVATION_PROGRESS.value]
            >= previous_neurons[:, Data.ACTIVATION_THRESHOLD.value]
        )
        activation_ready = updated_neurons[:, Data.ACTIVATION_WARMUP.value] >= 1
        activated = activation_threshold_reached & activation_ready
        self.log.activations = activated
        absolute_activation_mask = torch.zeros((self.neurons.size(0),), dtype=torch.bool)
        absolute_activation_mask[indices[activated]] = True
        self.log.io_activations = dict(zip(
            self.io_index_map.keys(), absolute_activation_mask[list(self.io_index_map.values())]))

        # process the current state of firing neurons and non-firing neurons to get state changes
        activated_state_update = self.genome.activation_transform(previous_neurons[activated, Data.STATE.value])
        passive_state_update = self.genome.passive_transform(previous_neurons[~activated, Data.STATE.value])
        activated_state_update[:, Data.TRANSFORM_INCREMENTED.value] = torch.tanh(
            activated_state_update[:, Data.TRANSFORM_INCREMENTED.value]
        )
        passive_state_update[:, Data.TRANSFORM_INCREMENTED.value] = torch.tanh(
            passive_state_update[:, Data.TRANSFORM_INCREMENTED.value]
        )

        # print('before', updated_neurons[~activated, Data.LATENT_STATE.value])
        # set activation warmup to 0 for neurons that have just fired
        updated_neurons[activated, Data.ACTIVATION_WARMUP.value] = 0
        # increment activation_warmup, cell_damage, and mitosis_stage
        updated_neurons[activated, Data.INCREMENTED_PARAMETERS.value] += (
            activated_state_update[:, Data.TRANSFORM_INCREMENTED.value])
        updated_neurons[~activated, Data.INCREMENTED_PARAMETERS.value] += (
            passive_state_update[:, Data.TRANSFORM_INCREMENTED.value])
        # print('after2', updated_neurons[~activated, Data.LATENT_STATE.value])

        updated_neurons[:, Data.INCREMENTED_PARAMETERS.value] = (
            updated_neurons[:, Data.INCREMENTED_PARAMETERS.value].clamp_min(0))
        # print('after1', updated_neurons[~activated, Data.LATENT_STATE.value])
        # set the updates to the latent state
        updated_neurons[activated, Data.LATENT_STATE.value] = activated_state_update[:, Data.TRANSFORM_LATENT.value]
        updated_neurons[~activated, Data.LATENT_STATE.value] = passive_state_update[:, Data.TRANSFORM_LATENT.value]
        # if we don't normalize the latent, it will explode in magnitude since it is technically recurrent over time
        updated_neurons[:, Data.LATENT_STATE.value] = F.normalize(updated_neurons[:, Data.LATENT_STATE.value], dim=1)
        # print('after', updated_neurons[~activated, Data.LATENT_STATE.value])

        # send signals to destination neurons
        if not activated.any():
            return

        signals = connectivity * previous_neurons[:, Data.SIGNAL_STRENGTH.value]
        signals[~activated] = 0

        cumulative_signals = signals.sum(dim=0)
        updated_neurons[:, Data.ACTIVATION_PROGRESS.value] += cumulative_signals

    def _handle_hormones(self, previous_neurons: Tensor, updated_neurons: Tensor, distances: Tensor):
        # decay hormones
        updated_neurons[:, Data.HORMONE_INFLUENCE.value] = updated_neurons[:, Data.HORMONE_INFLUENCE.value] * F.sigmoid(self.genome.hormone_decay)

        # absorb hormones
        # smooth falloff function: strength = max(log10(10 - distance * 9/range), 0)
        # print(distances.shape, previous_neurons[:, Data.HORMONE_RANGE.value].shape)
        hormone_strengths = torch.log10(
            10 - distances * (9 / previous_neurons[:, Data.HORMONE_RANGE.value])).clamp_min(0)  # (n, n)
        # will be nan if distance is greater than hormone range (log(x < 0) is nan), or if hormone range is 0
        # in both cases, the effective hormone strength should be 0
        hormone_strengths[hormone_strengths.isnan()] = 0
        hormones = previous_neurons[:, Data.HORMONE_EMISSION.value]
        scaled_hormones = hormone_strengths.unsqueeze(2) * hormones.unsqueeze(1)  # (n, n, h)
        hormone_absorption = scaled_hormones.sum(dim=0)
        # hormone_absorption = torch.einsum("ij,ik->jk", hormone_strengths, hormones)  # (n, 10)
        updated_neurons[:, Data.HORMONE_INFLUENCE.value] += hormone_absorption

    def _handle_life_and_death(self, updated_neurons: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor]:
        # initiate apoptosis
        # if self.device == torch.device('cuda'):
        #     torch.cuda.synchronize()
        is_dying = updated_neurons[:, Data.CELL_DAMAGE.value] >= 1
        self._deallocate_neurons(indices[is_dying])
        surviving_neurons = updated_neurons[~is_dying]
        surviving_indices = indices[~is_dying]

        # find dividing cells
        is_dividing = surviving_neurons[:, Data.MITOSIS_STAGE.value] >= 1
        # if is_dividing.any():
        #     print('Dividing')
        dividing_neurons = surviving_neurons[is_dividing]
        dividing_neurons[:, Data.MITOSIS_STAGE.value] = 0

        mitosis_results: Tensor = self.genome.mitosis_results(dividing_neurons[:, Data.STATE.value])
        parent_latent = mitosis_results[:, Data.MITOSIS_PARENT_LATENT.value]
        child_latent = mitosis_results[:, Data.MITOSIS_CHILD_LATENT.value]
        split_offset = F.normalize(mitosis_results[:, Data.MITOSIS_SPLIT_POSITION.value]) * (NEURON_DIAMETER / 2)

        child_position = dividing_neurons[:, Data.POSITION.value] + split_offset

        dividing_neurons[:, Data.LATENT_STATE.value] = parent_latent
        dividing_neurons[:, Data.CELL_DAMAGE.value] += F.sigmoid(
            torch.tensor([self.genome.mitosis_damage], device=self.device))
        surviving_neurons[is_dividing] = dividing_neurons

        # add new cells
        child_indices = self.add_neurons(child_position, child_latent)

        updated_neurons = torch.cat([surviving_neurons, self.neurons[child_indices]], dim=0)

        self.log.neuron_death_count = is_dying.sum().item()
        self.log.neuron_creation_count = is_dividing.sum().item()
        return (
            updated_neurons,
            torch.cat([surviving_indices, child_indices], dim=0)
        )

    def _handle_physics(
        self, neurons: Tensor, directions: Tensor, distances: Tensor, connectivity: Tensor
    ):

        lo, hi = self.genome.connection_range - self.genome.connection_pull_margin, self.genome.connection_range
        in_range_mask = (distances <= hi) & (distances >= lo)
        attraction_strength = torch.zeros_like(distances)
        attraction_strength[in_range_mask] = (distances[in_range_mask] - lo) / self.genome.connection_pull_margin
        attraction_strength *= connectivity * self.genome.connection_pull_strength
        # print(directions.shape, attraction_strength.shape)

        in_range_mask = distances < NEURON_DIAMETER
        repulsion_strength = torch.zeros_like(distances, device=self.device)
        repulsion_strength[in_range_mask] = (1 - distances[in_range_mask] / NEURON_DIAMETER).clamp_min(0)
        repulsion_strength *= self.neuron_repulsion

        net_strength = -attraction_strength + repulsion_strength

        cumulative_movements = (net_strength.unsqueeze(2) * directions).sum(dim=1)
        cumulative_movement_perturbed = torch.normal(cumulative_movements, std=0.001)
        neurons[:, Data.POSITION.value] += cumulative_movement_perturbed

    def _compute_connectivity(self, previous_neurons: Tensor, updated_neurons: Tensor, distances: Tensor) -> Tensor:
        # calculate connectivity
        in_range = distances <= self.genome.connection_range
        in_range_indices = torch.nonzero(in_range, as_tuple=False)
        sender_indices = in_range_indices[:, 0]
        receiver_indices = in_range_indices[:, 1]

        sender_states = previous_neurons[sender_indices, Data.STATE.value]
        receiver_states = previous_neurons[receiver_indices, Data.STATE.value]
        # print('ba', sender_states.shape, receiver_states.shape)
        connection_strengths = F.sigmoid(
            self.genome.connectivity_coefficient(torch.cat([sender_states, receiver_states], dim=1))
        ).squeeze(1)
        scoring_grid = torch.zeros_like(distances, device=self.device)
        # print(scoring_grid.shape, connection_strengths.shape, in_range_indices.shape)
        scoring_grid[in_range] = connection_strengths
        receptivity_scores = scoring_grid.sum(dim=0)
        emissivity_scores = scoring_grid.sum(dim=1)
        updated_neurons[:, Data.TOTAL_RECEPTIVITY.value] = receptivity_scores
        updated_neurons[:, Data.TOTAL_EMISSIVITY.value] = emissivity_scores

        connectivity_grid = torch.full_like(distances, fill_value=float('-inf'), device=self.device)
        connectivity_grid[in_range] = connection_strengths
        connectivity = connectivity_grid.softmax(dim=1) * scoring_grid  # (n, n)

        return connectivity

    def add_neurons(
        self,
        positions: Tensor,
        latent_states: Tensor,
        set_parameters: bool = True,
        variants: List[str] = None
    ) -> Tensor:
        """


        Cell damage, mitosis stage, total receptivity, total emissivity, activation progress,
        and hormone influence are initialized as zeros.

        :param positions:
        :param latent_states:
        :param set_parameters:
        :param variants:
        :return: the allocated neuron indices
        """
        if variants is None:
            variants = [None] * positions.size(0)

        positions, latent_states = positions.to(self.device), latent_states.to(self.device)
        neuron_indices = self._allocate_neurons(positions.size(0))

        self.neurons[neuron_indices, :] = 0
        self.neurons[neuron_indices, Data.POSITION.value] = positions
        self.neurons[neuron_indices, Data.LATENT_STATE.value] = latent_states
        self.io_labels[neuron_indices] = torch.tensor(
            [0 if label is None else self.io_label_to_int[label] for label in variants],
            dtype=torch.uint8,
            device=self.device
        )

        if set_parameters:
            parameters = self.genome.derive_parameters_from_state(self.neurons[neuron_indices, Data.STATE.value])
            self.neurons[neuron_indices, Data.DERIVED_PARAMETERS.value] = parameters

        return neuron_indices

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
            self.neurons = torch.cat((self.neurons, torch.zeros((extension_size, NEURON_DATA_DIM), device=self.device)))
            self.dead_neurons_indices.extend(
                range(current_neuron_buffer_size, current_neuron_buffer_size + extension_size))
            self.io_labels = torch.cat((
                self.io_labels,
                torch.zeros(extension_size, device=self.device, dtype=torch.uint8)
            ))
        allocation = self.dead_neurons_indices[:neuron_count]
        self.dead_neurons_indices = self.dead_neurons_indices[neuron_count:]
        self.living_neuron_indices.extend(allocation)
        return torch.tensor(allocation, dtype=torch.int, device=self.device)

    def _deallocate_neurons(self, indices: Tensor):
        indices = set(indices.tolist())
        self.living_neuron_indices = list(filter(lambda i: i not in indices, self.living_neuron_indices))
        for i in indices:
            self.io_index_map.pop(i)
        self.dead_neurons_indices.extend(indices)

    def stimulate_input_neurons(self, stimulus_map: Dict[int, float]):
        indices = []
        stimuli = []
        for io_index, stimulus_value in stimulus_map.items():
            if io_index not in self.io_index_map:
                continue
            indices.append(self.io_index_map[io_index])
            stimuli.append(stimulus_value)
        self.neurons[indices, Data.ACTIVATION_PROGRESS.value] += torch.tensor(stimuli, device=self.device)


class StepLog:
    def __init__(self):
        self.neuron_death_count = 0
        self.neuron_creation_count = 0
        self.neuron_count = 0
        self.neuron_positions = None
        self.connectivity = None
        self.activations = None
        self.io_activations: Dict[int, bool] = {}
