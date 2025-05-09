from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neuron.genome import Genome, mutate_genome, init_genome, InitialNeuronConfiguration
from neuron.specimen import Specimen

# import modal

# app = modal.App("my-project")
# image_with_source = modal.Image.debian_slim().pip_install('torch', 'numpy', 'tqdm').add_local_python_source("neuron")
#
#
# @app.function(image=image_with_source, gpu='T4')
def simulate_generation(
    genomes: List[Genome],
    survival_rate: float = 0.1,
    iterations: int = 600
) -> List[Tuple[Genome, float]]:
    survivor_count = int(len(genomes) * survival_rate)
    scores = []
    # def run(genome):
    #     torch.cuda.empty_cache()
    #     score = simulate_run(create_specimen(genome), iterations)
    #     return score
    #     # scores.append(score)
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(run, genome) for genome in genomes]
    #     scores = []
    #     for future in tqdm(as_completed(futures), total=len(futures), desc='Simulating generation'):
    #         scores.append(future.result())
    #     # scores = tqdm(executor.map(run, genomes), desc='Simulating generation', total=len(genomes))
    for genome in tqdm(genomes, desc='Simulating generation'):
        torch.cuda.empty_cache()
        score = simulate_run(create_specimen(genome), iterations)
        scores.append(score)
    # scores = list(scores)
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


# @app.local_entrypoint()
def main():
    torch.set_grad_enabled(False)
    generation_size = 100
    population = reproduce([init_genome()], generation_size)
    for generation in range(10):
        torch.cuda.empty_cache()
        results = simulate_generation(population, survival_rate=0.1, iterations=600)
        scores = [score for _, score in results]
        population = reproduce([genome for genome, _ in results], generation_size)
        print(f'Average score for generation {generation} survivors: {sum(scores) / len(scores)}')
    population[0].save('genome.pt')


if __name__ == '__main__':
    main()
