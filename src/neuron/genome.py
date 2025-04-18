from dataclasses import dataclass

import torch
from torch import Tensor, nn

from neuron.data_structure import LATENT_DIM, STATE_SIZE, HIDDEN_DIM, DERIVED_PARAMETERS_SIZE, TRANSFORM_SIZE


@dataclass
class Genome:
    # initial latent state for the first neuron(s)
    pluripotent_latent_state: Tensor
    # takes in internal state, outputs derived parameters
    derive_parameters_from_state: nn.Module
    # takes in internal state, outputs new latent state and state updates
    passive_transform: nn.Module
    # takes in internal state, outputs new latent state and state updates
    activation_transform: nn.Module
    # values that have sigmoid applied and are multiplied against hormone_influence each step
    hormone_decay: Tensor
    # takes in internal state pair and relative position (emitter and receiver and emitter to receiver),
    # outputs receptivity of latter [0,1]
    connectivity_coefficient: nn.Module
    # takes in internal state, outputs new latent state and daughter latent state and mitosis direction
    mitosis_results: nn.Module
    # sigmoid(mitosis_health_penalty) / 2 + 0.5 is the cell damage increment upon mitosis
    mitosis_damage: float
    connection_range: float
    # value that has sigmoid applied and is multiplied against activation_progress each step
    activation_decay: float


def init_genome():
    pluripotent_latent_state = torch.zeros(LATENT_DIM)
    derive_parameters_from_state = init_derive_parameters_from_state()


def init_derive_parameters_from_state():
    # STATE_SIZE -> DERIVED_PARAMETERS_SIZE
    # desired initial output: [0.5, 1.0, <0.0 segment>, 10.0]
    initial_output = torch.zeros(DERIVED_PARAMETERS_SIZE)
    initial_output[0] = 0.5  # activation threshold
    initial_output[1] = 1.0  # signal strength
    initial_output[-1] = 10.0  # hormone range

    layer1 = empty_linear(STATE_SIZE, HIDDEN_DIM)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer4 = empty_linear(HIDDEN_DIM, DERIVED_PARAMETERS_SIZE)
    layer4.bias.copy_(initial_output)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def init_passive_transform():
    # STATE_SIZE -> TRANSFORM_SIZE
    # desired initial output: [0.1, -0.1, 0.02, <identity of latent>]
    assert HIDDEN_DIM >= STATE_SIZE
    input_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE))
    input_matrix[:STATE_SIZE, :].copy_(torch.eye(STATE_SIZE))  # put an identity matrix into the top of the weights
    hidden_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    hidden_matrix[:STATE_SIZE, :STATE_SIZE].copy_(torch.eye(STATE_SIZE))
    output_matrix = torch.zeros((TRANSFORM_SIZE, HIDDEN_DIM))
    output_matrix[-STATE_SIZE:, :STATE_SIZE].copy_(torch.eye(STATE_SIZE))
    # activation warmup, cell damage, mitosis stage
    output_bias = torch.zeros(TRANSFORM_SIZE)
    output_bias[:3] = torch.Tensor([0.1, -0.1, 0.02])

    layer1 = empty_linear(STATE_SIZE, HIDDEN_DIM)
    layer1.weight.copy_(input_matrix)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer2.weight.copy_(hidden_matrix)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3.weight.copy_(hidden_matrix)
    layer4 = empty_linear(HIDDEN_DIM, TRANSFORM_SIZE)
    layer4.weight.copy_(output_matrix)
    layer4.bias.copy_(output_bias)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def init_connectivity_coefficient():
    # STATE_SIZE + STATE_SIZE -> 1
    # first rotate the first state's rows down 1
    # next, add all and subtract bias of 1
    # sum last layer
    roll_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE * 2))
    roll_matrix[:STATE_SIZE * 2, :] = torch.eye(STATE_SIZE * 2)
    roll_matrix[:STATE_SIZE, :STATE_SIZE] = roll_matrix[:STATE_SIZE, :STATE_SIZE].roll(1, 0)
    hidden_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    hidden_matrix[:STATE_SIZE * 2, :STATE_SIZE * 2].copy_(torch.eye(STATE_SIZE))
    add_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    add_matrix[:STATE_SIZE, :STATE_SIZE] = torch.eye(STATE_SIZE)
    add_matrix[:STATE_SIZE, STATE_SIZE:STATE_SIZE * 2] = torch.eye(STATE_SIZE)
    sum_matrix = torch.ones((1, HIDDEN_DIM))

    layer1 = empty_linear(STATE_SIZE * 2, HIDDEN_DIM)
    layer1.weight.copy_(roll_matrix)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer2.weight.copy_(hidden_matrix)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3.weight.copy_(add_matrix)
    layer3.bias.copy_(-1.0)
    layer4 = empty_linear(HIDDEN_DIM, 1)
    layer4.weight.copy_(sum_matrix)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def empty_linear(*shape):
    layer = nn.Linear(*shape)
    layer.weight.zero_()
    layer.bias.zero_()
    return layer
