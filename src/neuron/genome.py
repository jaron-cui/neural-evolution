import math
from dataclasses import dataclass

import copy
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn

from neuron.data_structure import LATENT_DIM, STATE_SIZE, HIDDEN_DIM, DERIVED_PARAMETERS_SIZE, TRANSFORM_SIZE, Data, \
    MITOSIS_SIZE, HORMONE_DIM


@dataclass
class InitialNeuronConfiguration:
    differentiation: Optional[str]
    start_position: Tensor


@dataclass
class Genome:
    # initial latent state for the first neuron(s)
    pluripotent_latent_state: Tensor
    differentiated_latent_states: Dict[str, Tensor]
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

    def to(self, device: torch.device):
        return Genome(
            self.pluripotent_latent_state.to(device),
            {key: value.to(device) for key, value in self.differentiated_latent_states.items()},
            self.derive_parameters_from_state.to(device),
            self.passive_transform.to(device),
            self.activation_transform.to(device),
            self.hormone_decay.to(device),
            self.connectivity_coefficient.to(device),
            self.mitosis_results.to(device),
            self.mitosis_damage,
            self.connection_range,
            self.connection_pull_margin,
            self.connection_pull_strength,
            self.activation_decay
        )


class MinibatchWrapper(nn.Module):
    def __init__(self, module: nn.Module, batch_size: int):
        super().__init__()
        self.module = module
        self.batch_size = batch_size

    def forward(self, x: Tensor):
        if x.size(0) <= self.batch_size:
            return self.module(x)
        results = []
        for batch_index in range(math.ceil(x.size(0) / self.batch_size)):
            result = self.module(x[batch_index * self.batch_size:(batch_index + 1) * self.batch_size])
            results.append(result)
        return torch.cat(results, dim=0)


def mutate_genome(genome: Genome) -> Genome:
    std = 0.01
    torch.normal(genome.pluripotent_latent_state, std)
    return Genome(
        pluripotent_latent_state=torch.normal(genome.pluripotent_latent_state, std),
        differentiated_latent_states={
            key: torch.normal(value, std) for key, value in genome.differentiated_latent_states.items()},
        derive_parameters_from_state=mutate_network(genome.derive_parameters_from_state, std),
        passive_transform=mutate_network(genome.passive_transform, std),
        activation_transform=mutate_network(genome.activation_transform, std),
        hormone_decay=torch.normal(genome.hormone_decay, std),
        connectivity_coefficient=mutate_network(genome.connectivity_coefficient, std),
        mitosis_results=mutate_network(genome.mitosis_results, std),
        mitosis_damage=torch.normal(Tensor([genome.mitosis_damage]), std).item(),
        connection_range=torch.normal(Tensor([genome.connection_range]), std).item(),
        connection_pull_margin=torch.normal(Tensor([genome.connection_pull_margin]), std).item(),
        connection_pull_strength=torch.normal(Tensor([genome.connection_pull_strength]), std).item(),
        activation_decay=torch.normal(Tensor([genome.activation_decay]), std).item(),
    )


def mutate_network(network: nn.Module, std: float) -> nn.Module:
    cloned = copy.deepcopy(network)
    with torch.no_grad():
        for parameter in cloned.parameters():
            parameter.data.copy_(torch.normal(parameter.data, std))
    return cloned


def init_start_configuration():
    return [InitialNeuronConfiguration(None, torch.tensor([0.0, 0.0, 0.0]))]


def init_genome(differentiations: Iterable[str] = None):
    if differentiations is None:
        differentiations = []
    initial_latent_state = init_pluripotent_latent_state()
    genome = Genome(
        pluripotent_latent_state=initial_latent_state,
        differentiated_latent_states={
            differentiation: initial_latent_state.clone() for differentiation in differentiations
        },
        derive_parameters_from_state=init_derive_parameters_from_state(),
        passive_transform=init_passive_transform(),
        activation_transform=init_passive_transform(),
        hormone_decay=torch.full((HORMONE_DIM,), fill_value=2.0),
        connectivity_coefficient=init_connectivity_coefficient(),
        mitosis_results=init_mitosis_results(),
        mitosis_damage=0.5,
        connection_range=5,
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

    return nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3, nn.ReLU(inplace=True), layer4)


def init_passive_transform():
    # STATE_SIZE -> TRANSFORM_SIZE
    # desired initial output: [0.1, -0.1, 0.02, <identity of latent>]
    assert HIDDEN_DIM >= STATE_SIZE
    input_matrix = torch.zeros((HIDDEN_DIM, STATE_SIZE))
    # print(Data.TRANSFORM_LATENT.value)
    input_matrix[:LATENT_DIM, Data.TRANSFORM_LATENT.value].copy_(torch.eye(LATENT_DIM))  # put an identity matrix into the top of the weights
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
    # print(input_matrix)
    # print(hidden_matrix)
    # print(output_matrix)
    # class M(nn.Module):
    #     def forward(self, x):
    #         if x.numel() == 0:
    #             return nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3, nn.ReLU(inplace=True), layer4)(x)
    #         out = layer1(x)
    #         print(out.max().item())
    #         out = nn.functional.relu(out)
    #         out = layer2(out)
    #         print(out.max().item())
    #         out = nn.functional.relu(out)
    #         out = layer3(out)
    #         print(out.max().item())
    #         out = nn.functional.relu(out)
    #         out = layer4(out)
    #         print(out.max().item())
    #         return out
    return nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3, nn.ReLU(inplace=True), layer4)


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

    return MinibatchWrapper(nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3, nn.ReLU(inplace=True), layer4), 128)


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

    return nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3, nn.ReLU(inplace=True), layer4)


def empty_linear(*shape):
    layer = nn.Linear(*shape)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    return layer
