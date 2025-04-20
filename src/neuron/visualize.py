import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuron.data_structure import Data
from neuron.genome import init_genome
from neuron.specimen import Specimen


def draw_specimen(specimen: Specimen):
    scale = 20
    neurons = specimen.neurons[torch.tensor(specimen.living_neuron_indices, dtype=torch.int)]
    background = torch.zeros((scale * 50, scale * 50), dtype=torch.float)
    positions = (neurons[:, Data.POSITION.value] * scale).to(dtype=torch.int)
    for yo, xo in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        background[positions[:, 1] + yo, positions[:, 0] + xo] = 1.0
    # for neuron in neurons:
    #     x, y, z = (neuron[Data.POSITION.value] * scale).to(dtype=torch.int)
    #     background[y - 2:y + 2, x - 2:x + 2] = 1.0

    image = background.unsqueeze(2).repeat((1, 1, 3))
    cv2.imshow('image', image.numpy())
    # plt.imshow(image)
    # plt.show()


# genome = init_genome()
# specimen = Specimen(genome)
# specimen.add_neurons(
#     torch.Tensor([[20, 20, 0], [22, 20, 0]]),
#     genome.pluripotent_latent_state.unsqueeze(0).repeat((2, 1))
# )

# start = time.time()
# for i in range(350):
#     # print('Stepping')
#     specimen.step()
#     # print(len(specimen.living_neuron_indices))
#     # print(i)
#     # print('Stepped', time.time() - start)
#     # draw_specimen(specimen)
#     # key = cv2.waitKey(0)
#     # if key == ord('q'):
#     #     break
# print(time.time() - start)
# cv2.destroyAllWindows()

# neurons = np.load('before_bad.npy')
# indices = np.load('indices.npy')
# specimen.neurons = torch.from_numpy(neurons)
# for i in range(len(neurons)):
#     if specimen.neurons[i].isnan().any():
#         print(i)
#         print(specimen.neurons[i])
#         exit(0)
# print('n', specimen.neurons[0].isnan())
# specimen.living_neuron_indices = list(indices)
# specimen.dead_neurons_indices = [i for i in range(len(neurons)) if i not in indices]
# print('sim')
# print(specimen.neurons[torch.tensor(specimen.living_neuron_indices, dtype=torch.int)])
# print('step')
# specimen.step()
# print(specimen.neurons[torch.tensor(specimen.living_neuron_indices, dtype=torch.int)])
