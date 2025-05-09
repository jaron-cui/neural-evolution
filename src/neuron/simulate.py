import time
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neuron.genome import Genome, mutate_genome, init_genome, InitialNeuronConfiguration
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
        # closeness to desired neuron count
        score += 1 - (abs(len(specimen.living_neuron_indices) - optimal_neuron_count) / optimal_neuron_count)
        # penalize neuron deaths
        score -= (specimen.log.neuron_death_count / optimal_neuron_count)**2 * 2
        if specimen.log.neuron_count > cancer_threshold or specimen.log.neuron_count == 0:
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
    specimen = Specimen(genome, [
        InitialNeuronConfiguration(None, torch.tensor([20.0, 20.0, 0.0])),
        InitialNeuronConfiguration(None, torch.tensor([22.0, 20.0, 0.0]))
    ])
    return specimen


def main():
    torch.set_grad_enabled(False)
    generation_size = 100
    population = reproduce([init_genome()], generation_size)
    for generation in range(1):
        torch.cuda.empty_cache()
        results = simulate_generation(population, survival_rate=0.1, iterations=600)
        scores = [score for _, score in results]
        population = reproduce([genome for genome, _ in results], generation_size)
        print(f'Average score for generation {generation} survivors: {sum(scores) / len(scores)}')
    population[0].save('genome.pt')


if __name__ == '__main__':
    main()
