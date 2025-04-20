from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neuron.genome import Genome, mutate_genome, init_genome
from neuron.specimen import Specimen


def simulate_generation(
    genomes: List[Genome],
    survival_rate: float = 0.1,
    iterations: int = 600
) -> List[Tuple[Genome, float]]:
    survivor_count = int(len(genomes) * survival_rate)
    scores = []
    for genome in tqdm(genomes, desc='Simulating generation'):
        torch.cuda.empty_cache()
        score = simulate_run(create_specimen(genome), iterations)
        scores.append(score)
    survivor_indices = np.argsort(scores)[-survivor_count:]
    return [(genomes[i], scores[i]) for i in survivor_indices]


def simulate_run(specimen: Specimen, iterations: int) -> float:
    cancer_threshold = 200
    optimal_neuron_count = 100

    score = 0
    for i in range(iterations):
        specimen.step()
        score += 1 - (abs(len(specimen.living_neuron_indices) - optimal_neuron_count) / optimal_neuron_count)
        if len(specimen.living_neuron_indices) > cancer_threshold:
            return score
    return score


def reproduce(genomes: List[Genome], target_count: int) -> List[Genome]:
    children = []
    while True:
        for genome in genomes:
            if len(children) >= target_count:
                return children
            child_genome = mutate_genome(genome)
            children.append(child_genome)


def create_specimen(genome: Genome) -> Specimen:
    specimen = Specimen(genome)
    specimen.add_neurons(
        torch.Tensor([[20, 20, 0], [22, 20, 0]]),
        genome.pluripotent_latent_state.unsqueeze(0).repeat((2, 1))
    )
    return specimen


def main():
    generation_size = 100
    population = reproduce([init_genome()], generation_size)
    for generation in range(10):
        results = simulate_generation(population, survival_rate=0.1, iterations=600)
        scores = [score for _, score in results]
        population = reproduce([genome for genome, _ in results], generation_size)
        print(f'Average score for generation {generation} survivors: {sum(scores) / len(scores)}')


if __name__ == '__main__':
    main()
