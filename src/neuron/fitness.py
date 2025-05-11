from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributions as dist

from neuron.specimen import Specimen


class Criteria(ABC):
    @abstractmethod
    def accumulate(self, specimen: Specimen):
        """
        Accumulate information from the specimen at a given timestep in order to assess fitness.

        :param specimen: a specimen at a given timestep
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_fitness_score(self) -> float:
        """
        Calculate the overall fitness score after the specimen simulation has ended.

        :return: a fitness score, where a perfect score ought to be equal to the number of time steps
        """
        raise NotImplementedError()


class TargetNeuronCountCriteria(Criteria):
    """
    Reward the specimen for consistently having a total neuron count close to a set number.
    """
    def __init__(self, target_neuron_count: int):
        super().__init__()
        self.target_neuron_count = target_neuron_count
        self.total_fitness_score = 0

    def accumulate(self, specimen: Specimen):
        diff = abs(len(specimen.living_neuron_indices) - self.target_neuron_count)
        normalized_timestep_score = 1 - (diff / self.target_neuron_count)
        self.total_fitness_score += normalized_timestep_score

    def calculate_fitness_score(self) -> float:
        return self.total_fitness_score


class NeuronSurvivalCriteria(Criteria):
    """
    Penalize the specimen for neuron deaths.
    """
    def __init__(self, target_neuron_count: int):
        super().__init__()
        self.target_neuron_count = target_neuron_count
        self.total_fitness_score = 0

    def accumulate(self, specimen: Specimen):
        penalty = (specimen.log.neuron_death_count / self.target_neuron_count) ** 2 * 2
        self.total_fitness_score -= penalty

    def calculate_fitness_score(self) -> float:
        return self.total_fitness_score


class SignalMatchingCriteria(Criteria):
    def __init__(self, pattern: List[int], output_neuron_index: int, pulse_std: float):
        super().__init__()
        self.pattern = pattern
        self.output_neuron_index = output_neuron_index
        self.pulse_std = pulse_std
        self.timestep = 0
        self.recorded_pattern = []

    def accumulate(self, specimen: Specimen):
        self.timestep += 1
        activation = specimen.log.io_activations.get(self.output_neuron_index, False)
        if activation:
            self.recorded_pattern.append(self.timestep)

    def calculate_fitness_score(self) -> float:
        if len(self.recorded_pattern) == 0:
            return -self.timestep
        target_offset, recorded_offset = min(self.pattern), min(self.recorded_pattern)
        target = torch.tensor([frame - target_offset for frame in self.pattern])
        recorded = torch.tensor([frame - recorded_offset for frame in self.recorded_pattern])

        normal = dist.Normal(target, self.pulse_std)
        peak_probability = dist.Normal(0, self.pulse_std).log_prob(torch.zeros(1)).exp().item()
        max_probabilities, _ = normal.log_prob(recorded.unsqueeze(0).repeat((target.size(0), 1)).T).max(dim=0)
        normalized_max_probabilities = max_probabilities.exp() / peak_probability

        score = (normalized_max_probabilities - 0.2).sum()

        return score


class TargetActivationRateCriteria(Criteria):
    """
    Reward the specimen for consistently having a total neuron count close to a set number.
    """
    def __init__(self, target_activation_rate: float):
        super().__init__()
        self.target_activation_rate = target_activation_rate
        self.time_steps = 0
        self.total_activations = 0

    def accumulate(self, specimen: Specimen):
        self.time_steps += 1
        self.total_activations += specimen.log.activations.detach().cpu().sum()

    def calculate_fitness_score(self) -> float:
        return 1 - abs(self.total_activations / self.time_steps - self.target_activation_rate)

