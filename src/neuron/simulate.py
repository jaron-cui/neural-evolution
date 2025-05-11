import logging
import sys
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neuron.fitness import TargetNeuronCountCriteria, NeuronSurvivalCriteria, SignalMatchingCriteria, \
    TargetActivationRateCriteria
from neuron.genome import Genome, mutate_genome, init_genome, InitialNeuronConfiguration
from neuron.specimen import Specimen
from utils import TrainingRecord


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
    iteration_death_penalty = 10
    pattern = [150, 170, 180, 185, 200, 250, 260, 265, 275, 290, 300, 310, 390, 450, 460, 480, 500, 550, 555, 560, 565, 575, 590]

    criteria = [
        TargetNeuronCountCriteria(optimal_neuron_count),
        NeuronSurvivalCriteria(optimal_neuron_count),
        SignalMatchingCriteria(pattern, 1, 5),
        # TargetActivationRateCriteria(0.1)
    ]
    auxiliary_score = 0
    for i in range(iterations):
        if i in pattern:
            specimen.stimulate_input_neurons({0: 1.0})
        specimen.step()
        for criterion in criteria:
            criterion.accumulate(specimen)
        if specimen.log.neuron_count > cancer_threshold or specimen.log.neuron_count == 0 or 0 not in specimen.io_index_map or 1 not in specimen.io_index_map.values():
            death_penalty = iteration_death_penalty * (iterations - i)
            auxiliary_score -= death_penalty
            break
    return sum([criterion.calculate_fitness_score() for criterion in criteria]) + auxiliary_score


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
        InitialNeuronConfiguration('input', torch.tensor([20.0, 20.0, 0.0])),
        InitialNeuronConfiguration('output', torch.tensor([26.0, 20.0, 0.0]))
    ])
    return specimen


def train(
    generations: int,
    generation_size: int,
    specimen_lifespan: int,
    save_every: int = 1,
    resume_training_run_from: str = None
):
    if resume_training_run_from is None:
        training_record = TrainingRecord('checkpoints')
        population = reproduce([init_genome(['input', 'output'])], generation_size)
    else:
        training_record, survivors = TrainingRecord.resume_from(resume_training_run_from)
        population = reproduce(survivors, generation_size)
        logging.info('----------------------------------------    /    ----------------------------------------')
        logging.info(f'Resuming evolution from `{resume_training_run_from}` for {generations} additional generations.')

    torch.set_grad_enabled(False)
    for generation in range(training_record.last_epoch + 1, training_record.last_epoch + 1 + generations):
        torch.cuda.empty_cache()
        logging.info(f'Simulating generation {generation} with {len(population)} specimens.')
        results = simulate_generation(population, survival_rate=0.1, iterations=specimen_lifespan)
        scores = [score for _, score in results]
        survivors = [genome for genome, _ in results]
        population = reproduce(survivors, generation_size)
        logging.info(f'Average score for generation {generation} survivors: {sum(scores) / len(scores)}')
        if generation % save_every == 0:
            training_record.save_checkpoint(survivors, generation)


# @app.local_entrypoint()
def main():
    train(10, 100, 600, resume_training_run_from='checkpoints/2025-05-11/11-57-23')
    # training_record = TrainingRecord('checkpoints')
    # save_every = 1
    #
    # torch.set_grad_enabled(False)
    # generation_size = 100
    # population = reproduce([init_genome()], generation_size)
    # for generation in range(10):
    #     torch.cuda.empty_cache()
    #     results = simulate_generation(population, survival_rate=0.1, iterations=600)
    #     scores = [score for _, score in results]
    #     survivors = [genome for genome, _ in results]
    #     population = reproduce(survivors, generation_size)
    #     print(f'Average score for generation {generation} survivors: {sum(scores) / len(scores)}')
    #     if generation % save_every == 0:
    #         training_record.save_checkpoint(survivors, generation)
    # population[0].save('genome.pt')


if __name__ == '__main__':
    main()
