from dataclasses import dataclass

import torch
from torch import Tensor, nn

from neuron.data_structure import LATENT_DIM, STATE_SIZE, HIDDEN_DIM, DERIVED_PARAMETERS_SIZE, TRANSFORM_SIZE, Data, \
    MITOSIS_SIZE, HORMONE_DIM


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
    connection_pull_margin: float
    connection_pull_strength: float
    # value that has sigmoid applied and is multiplied against activation_progress each step
    activation_decay: float


def init_genome():
    genome = Genome(
        pluripotent_latent_state=init_pluripotent_latent_state(),
        derive_parameters_from_state=init_derive_parameters_from_state(),
        passive_transform=init_passive_transform(),
        activation_transform=init_passive_transform(),
        hormone_decay=torch.full((HORMONE_DIM,), fill_value=0.9),
        connectivity_coefficient=init_connectivity_coefficient(),
        mitosis_results=init_mitosis_results(),
        mitosis_damage=0.5,
        connection_range=2,
        connection_pull_margin=1.0,
        connection_pull_strength=0.1,
        activation_decay=0.9
    )
    return genome


def init_pluripotent_latent_state():
    pluripotent_latent_state = torch.zeros(LATENT_DIM)
    pluripotent_latent_state[0] = 1
    return pluripotent_latent_state


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
    with torch.no_grad():
        layer4.bias.copy_(initial_output)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def init_passive_transform():
    # STATE_SIZE -> TRANSFORM_SIZE
    # desired initial output: [0.1, -0.1, 0.02, <identity of latent>]
    assert HIDDEN_DIM >= STATE_SIZE
    input_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE))
    input_matrix[:LATENT_DIM, :LATENT_DIM].copy_(torch.eye(LATENT_DIM))  # put an identity matrix into the top of the weights
    hidden_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    hidden_matrix[:LATENT_DIM, :LATENT_DIM].copy_(torch.eye(LATENT_DIM))
    output_matrix = torch.zeros((TRANSFORM_SIZE, HIDDEN_DIM))
    output_matrix[-LATENT_DIM:, :LATENT_DIM].copy_(torch.eye(LATENT_DIM))
    # activation warmup, cell damage, mitosis stage
    output_bias = torch.zeros(TRANSFORM_SIZE)
    output_bias[:3] = torch.Tensor([0.1, -0.1, 0.02])

    layer1 = empty_linear(STATE_SIZE, HIDDEN_DIM)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer4 = empty_linear(HIDDEN_DIM, TRANSFORM_SIZE)
    with torch.no_grad():
        layer1.weight.copy_(input_matrix)
        layer2.weight.copy_(hidden_matrix)
        layer3.weight.copy_(hidden_matrix)
        layer4.weight.copy_(output_matrix)
        layer4.bias.copy_(output_bias)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def init_connectivity_coefficient():
    # STATE_SIZE + STATE_SIZE -> 1
    # first rotate the first state's rows down 1
    # next, add all and subtract bias of 1
    # sum last layer
    roll_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE * 2))
    roll_matrix[:LATENT_DIM, Data.TRANSFORM_LATENT.value] = torch.eye(LATENT_DIM).roll(1, 0)
    roll_matrix[LATENT_DIM:LATENT_DIM * 2, STATE_SIZE:][:, Data.TRANSFORM_LATENT.value] = torch.eye(LATENT_DIM)
    hidden_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    hidden_matrix[:LATENT_DIM * 2, :LATENT_DIM * 2] = torch.eye(LATENT_DIM * 2)
    add_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    add_matrix[:LATENT_DIM, :LATENT_DIM] = torch.eye(LATENT_DIM)
    add_matrix[:LATENT_DIM, LATENT_DIM:LATENT_DIM * 2] = torch.eye(LATENT_DIM)
    sum_matrix = torch.ones((1, HIDDEN_DIM))

    layer1 = empty_linear(STATE_SIZE * 2, HIDDEN_DIM)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer4 = empty_linear(HIDDEN_DIM, 1)
    with torch.no_grad():
        layer1.weight.copy_(roll_matrix)
        layer2.weight.copy_(hidden_matrix)
        layer3.weight.copy_(add_matrix)
        layer3.bias.copy_(-1.0)
        layer4.weight.copy_(sum_matrix)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def init_mitosis_results():
    # STATE_SIZE -> MITOSIS_SIZE
    assert HIDDEN_DIM >= LATENT_DIM * 2
    copy_and_roll_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE))
    copy_and_roll_matrix[:LATENT_DIM, Data.TRANSFORM_LATENT.value] = torch.eye(LATENT_DIM)
    copy_and_roll_matrix[LATENT_DIM:LATENT_DIM * 2, Data.TRANSFORM_LATENT.value] = torch.eye(LATENT_DIM).roll(1, 0)
    hidden_matrix = torch.zeros((HIDDEN_DIM, HIDDEN_DIM))
    hidden_matrix[:LATENT_DIM * 2, :LATENT_DIM * 2] = torch.eye(LATENT_DIM * 2)
    output_matrix = torch.zeros((MITOSIS_SIZE, HIDDEN_DIM))
    output_matrix[:LATENT_DIM * 2, :LATENT_DIM * 2] = torch.eye(LATENT_DIM * 2)

    layer1 = empty_linear(STATE_SIZE, HIDDEN_DIM)
    layer2 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer3 = empty_linear(HIDDEN_DIM, HIDDEN_DIM)
    layer4 = empty_linear(HIDDEN_DIM, MITOSIS_SIZE)
    with torch.no_grad():
        layer1.weight.copy_(copy_and_roll_matrix)
        layer2.weight.copy_(hidden_matrix)
        layer3.weight.copy_(hidden_matrix)
        layer4.weight.copy_(output_matrix)

    return nn.Sequential(layer1, nn.GELU(), layer2, nn.GELU(), layer3, nn.GELU(), layer4)


def empty_linear(*shape):
    layer = nn.Linear(*shape)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    return layer
