import time

import torch
import matplotlib.pyplot as plt

from neuron.data_structure import Data
from neuron.genome import init_genome
from neuron.specimen import Specimen


def draw_specimen(specimen: Specimen):
    scale = 20
    neurons = specimen.neurons[torch.tensor(specimen.living_neuron_indices, dtype=torch.int)]
    background = torch.zeros((scale * 50, scale * 50), dtype=torch.float)
    for neuron in neurons:
        x, y, z = (neuron[Data.POSITION.value] * scale).to(dtype=torch.int)
        background[y - 5:y + 5, x - 5:x + 5] = 1.0

    image = background.unsqueeze(2).repeat((1, 1, 3))
    plt.imshow(image)
    plt.show()


genome = init_genome()
specimen = Specimen(genome)
specimen.add_neurons(
    torch.Tensor([[20, 20, 0], [22, 20, 0]]),
    genome.pluripotent_latent_state.unsqueeze(0).repeat((2, 1))
)
start = time.time()
print('Stepping')
specimen.step()
print('Stepped', time.time() - start)
draw_specimen(specimen)
