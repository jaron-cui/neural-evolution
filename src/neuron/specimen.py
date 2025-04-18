from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from neuron.data_structure import NEURON_DATA_DIM, Data, NEURON_DIAMETER
from neuron.genome import Genome


class Specimen:
    def __init__(self, genome: Genome, neuron_repulsion: float = 0.01):
        self.genome = genome
        self.neuron_repulsion = neuron_repulsion

        initial_neuron_buffer_size = 16
        self.living_neuron_indices = []
        self.dead_neurons_indices = list(range(initial_neuron_buffer_size))
        self.neurons = torch.zeros((initial_neuron_buffer_size, NEURON_DATA_DIM))

    def step(self):
        indices = torch.tensor(self.living_neuron_indices, dtype=torch.int)
        previous_neurons = self.neurons[indices]
        updated_neurons = previous_neurons.clone()

        positions = previous_neurons[: Data.POSITION.value]
        diffs = positions[:, None, :] - positions[None, :, :]  # shape: (n, n, 3)
        distances = torch.norm(diffs, dim=-1)  # shape: (n, n)
        directions = diffs / distances
        directions[directions.isnan()] = 0

        connectivity = self._compute_connectivity(previous_neurons, updated_neurons, distances)
        # update neuron positions
        self._handle_physics(updated_neurons, diffs, directions, distances, connectivity)
        # handle hormone emission, absorption, and decay
        self._handle_hormones(previous_neurons, updated_neurons, distances)
        # handle firing and passive neuron state updates
        self._handle_state_transform_updates(previous_neurons, updated_neurons, indices, connectivity)
        # handle cell death and division
        updated_neurons, indices = self._handle_life_and_death(updated_neurons, indices)
        # update derived parameters
        derived_parameters = self.genome.derive_parameters_from_state(updated_neurons[:, Data.STATE.value])
        updated_neurons[:, Data.DERIVED_PARAMETERS.value] = derived_parameters

    def _handle_state_transform_updates(
        self, previous_neurons: Tensor, updated_neurons: Tensor, indices: Tensor, connectivity: Tensor
    ):
        # decay ions
        updated_neurons[:, Data.ACTIVATION_PROGRESS.value].mul_(self.genome.activation_decay)

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
        updated_neurons[:, Data.INCREMENTED_PARAMETERS.value].relu_()
        # set the updates to the latent state
        updated_neurons[activated, Data.LATENT_STATE.value].copy_(
            activated_state_update[:, Data.TRANSFORM_LATENT.value])
        updated_neurons[~activated, Data.LATENT_STATE.value].copy_(passive_state_update[:, Data.TRANSFORM_LATENT.value])

        # send signals to destination neurons
        signals = (connectivity * previous_neurons[activated, Data.SIGNAL_STRENGTH.value]).reshape(-1)
        cumulative_signals = signals.sum(dim=0)
        updated_neurons[:, Data.ACTIVATION_PROGRESS.value].add_(cumulative_signals[indices])

    def _handle_hormones(self, previous_neurons: Tensor, updated_neurons: Tensor, distances: Tensor):
        # decay hormones
        updated_neurons[:, Data.HORMONE_INFLUENCE.value].mul_(self.genome.hormone_decay)

        # absorb hormones
        # smooth falloff function: strength = max(log10(10 - distance * 9/range), 0)
        hormone_strengths = torch.log10(
            10 - distances * (9 / previous_neurons[:, Data.HORMONE_RANGE.value])).relu_()  # (n, n)
        # scaled_hormones = hormone_strengths * previous_neurons[:, None, Data.HORMONE_EMISSION.value]  # (n, n, h)
        # hormone_absorption = scaled_hormones.sum(dim=0)
        hormones = previous_neurons[:, None, Data.HORMONE_EMISSION.value]
        hormone_absorption = torch.einsum("ij,ik->jk", hormone_strengths, hormones)  # (n, 10)
        updated_neurons[:, Data.HORMONE_INFLUENCE.value].add_(hormone_absorption)

    def _handle_life_and_death(self, updated_neurons: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor]:
        # initiate apoptosis
        dying_neurons = updated_neurons[:, Data.CELL_DAMAGE.value] >= 1
        self._deallocate_neurons(indices[dying_neurons])
        surviving_neurons = updated_neurons[~dying_neurons]
        surviving_indices = indices[~dying_neurons]

        # find dividing cells
        is_dividing = surviving_neurons[:, Data.MITOSIS_STAGE.value] >= 1
        dividing_neurons = surviving_neurons[is_dividing]

        mitosis_results: Tensor = self.genome.mitosis_results(dividing_neurons[:, Data.STATE.value])
        parent_latent = mitosis_results[Data.MITOSIS_PARENT_LATENT.value]
        child_latent = mitosis_results[Data.MITOSIS_CHILD_LATENT.value]
        split_offset = F.normalize(mitosis_results[Data.MITOSIS_SPLIT_POSITION.value]) * (NEURON_DIAMETER / 2)
        child_position = dividing_neurons[:, Data.POSITION.value] + split_offset

        dividing_neurons[:, Data.LATENT_STATE.value].copy_(parent_latent)
        child_indices = self.add_neurons(child_position, child_latent)

        updated_neurons = torch.cat([surviving_neurons, self.neurons[child_indices]], dim=0)
        updated_neurons[:, Data.CELL_DAMAGE.value].sub_(F.sigmoid(self.genome.mitosis_damage))

        return (
            updated_neurons,
            torch.cat([surviving_indices, child_indices], dim=0)
        )

    def _handle_physics(
        self, neurons: Tensor, diffs: Tensor, directions: Tensor, distances: Tensor, connectivity: Tensor
    ):
        lo, hi = self.genome.connection_range - self.genome.connection_pull_margin, self.genome.connection_range
        in_range_mask = (distances <= hi) & (distances >= lo)
        attraction_strength = torch.zeros_like(distances)
        attraction_strength[in_range_mask] = (distances[in_range_mask] - lo) / self.genome.connection_pull_margin
        attraction_strength.mul_(connectivity).mul_(self.genome.connection_pull_strength)
        attraction = -directions * attraction_strength

        repulsion = directions * NEURON_DIAMETER - diffs
        repulsion[repulsion.dot(directions) < 0].zero_()
        repulsion.mul_(self.neuron_repulsion / NEURON_DIAMETER)
        cumulative_movements = repulsion.sum(dim=1) + attraction.sum(dim=1)
        cumulative_movement_perturbed = torch.normal(cumulative_movements, std=0.02)
        neurons[:, Data.POSITION.value].add_(cumulative_movement_perturbed)

    def _compute_connectivity(self, previous_neurons: Tensor, updated_neurons: Tensor, distances: Tensor) -> Tensor:
        # calculate connectivity
        in_range = distances <= self.genome.connection_range
        in_range_indices = torch.nonzero(in_range, as_tuple=False)
        sender_indices = in_range_indices[:, 0]
        receiver_indices = in_range_indices[:, 1]

        sender_states = previous_neurons[sender_indices, Data.STATE.value]
        receiver_states = previous_neurons[receiver_indices, Data.STATE.value]
        connection_strengths = F.sigmoid(
            self.genome.connectivity_coefficient(torch.cat([sender_states, receiver_states], dim=1))
        )
        scoring_grid = torch.zeros_like(distances)
        scoring_grid[in_range_indices] = connection_strengths
        receptivity_scores = scoring_grid.sum(dim=0)
        emissivity_scores = scoring_grid.sum(dim=1)
        updated_neurons[:, Data.TOTAL_RECEPTIVITY.value] = receptivity_scores
        updated_neurons[:, Data.TOTAL_EMISSIVITY.value] = emissivity_scores

        connectivity_grid = torch.full_like(distances, fill_value=float('-inf'))
        connectivity_grid[in_range_indices] = connection_strengths
        connectivity = connectivity_grid.softmax(dim=1)  # (n, n)

        return connectivity

    def add_neurons(self, positions: Tensor, latent_states: Tensor, set_parameters: bool = True) -> Tensor:
        """


        Cell damage, mitosis stage, total receptivity, total emissivity, activation progress,
        and hormone influence are initialized as zeros.

        :param positions:
        :param latent_states:
        :param set_parameters:
        :return: the allocated neuron indices
        """
        neuron_indices = self._allocate_neurons(positions.size(0))

        self.neurons[neuron_indices, :].zero_()
        self.neurons[neuron_indices, Data.POSITION.value] = positions
        self.neurons[neuron_indices, Data.LATENT_STATE.value] = latent_states

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
