import random
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import torch

from neuron.data_structure import Data
from neuron.genome import Genome, InitialNeuronConfiguration, init_genome
from neuron.specimen import Specimen


class Plotter:
    def __init__(self):
        self.plotter = pv.Plotter(window_size=[800, 600])
        self.points_actor: Optional[pv.Actor] = None
        self.lines_actor: Optional[pv.Actor] = None

    def open_window(self):
        self.plotter.camera_position = 'iso'
        self.plotter.show(auto_close=False)

    def close_window(self):
        pass

    def set_state(
        self,
        coordinates: np.ndarray,
        point_colors: np.ndarray,
        connection_colors: np.ndarray
    ):
        point_cloud_mesh = pv.PolyData(coordinates)
        point_cloud_mesh['neuron_colors'] = point_colors
        lines_mesh = pv.PolyData(coordinates, lines=_line_segment_indices(coordinates.shape[0]))
        lines_mesh['line_colors'] = connection_colors
        if self.points_actor is None:
            self.points_actor = self.plotter.add_mesh(
                point_cloud_mesh,
                style='points',  # 'points' or 'spheres'
                render_points_as_spheres=True,
                point_size=10,
                scalars='neuron_colors',
                name='dynamic_points',
                show_scalar_bar=False,
                rgb=True
            )
            self.lines_actor = self.plotter.add_mesh(
                lines_mesh,
                scalars='line_colors',
                line_width=1,
                name='dynamic_lines',
                rgba=True
            )
        else:
            self.points_actor.mapper.dataset = point_cloud_mesh
            self.lines_actor.mapper.dataset = lines_mesh

        self.plotter.render()


def _line_segment_indices(m: int) -> np.ndarray:
    temp = np.expand_dims(np.arange(m, dtype=int), 1).repeat(m, axis=1)
    i = temp.flatten()
    j = temp.T.flatten()
    segments = np.stack([np.full_like(i, 2), i, j]).T
    lines = segments.flatten()
    return lines


def create_specimen(genome: Genome) -> Specimen:
    specimen = Specimen(genome, [
        InitialNeuronConfiguration(None, torch.tensor([0.0, 0.0, 0.0])),
        InitialNeuronConfiguration(None, torch.tensor([4.0, 0.0, 0.0]))
    ])
    return specimen


def visualization_information(specimen: Specimen) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = specimen.log.neuron_positions.detach().cpu().numpy() / 10
    activations = specimen.log.activations.detach().cpu().numpy()
    neuron_color = np.zeros((coordinates.shape[0], 3))
    neuron_color[:] = np.array([0.1, 0.1, 1.0])

    neuron_color[activations] = np.array([1.0, 1.0, 0.1])

    activations_matrix = np.zeros((coordinates.shape[0], coordinates.shape[0]), dtype=np.bool)
    activations_matrix[activations] = True

    connectivity = specimen.log.connectivity.detach().cpu().numpy().flatten()
    connection_opacity = np.clip(connectivity.flatten() * 0.1, 0.0, 1.0)
    connection_color = np.zeros((connectivity.shape[0], 4))
    connection_color[:] = np.array([0.6, 0.6, 0.85, 0.0])

    connection_color[:, -1] = connection_opacity
    connection_color[activations_matrix.T.flatten()] = np.array([1.0, 1.0, 0.1, 1.0])
    # cull bottom 90% connections
    connection_color[np.argsort(connectivity)[:-connection_opacity.shape[0] // 20], -1] = 0
    return coordinates, neuron_color, connection_color


def main():
    genome = torch.load('../../checkpoints/2025-05-11/11-57-23/generation_58_survivors.pt', weights_only=False)[0]
    # genome = init_genome()
    # print(genome.activation_decay, torch.nn.functional.sigmoid(torch.Tensor([genome.activation_decay])).item())
    specimen = create_specimen(genome)

    plotter = Plotter()

    def step():
        if random.random() < 0.01:
            specimen.stimulate_input_neurons({0: 1.0})
        specimen.step()
        plotter.set_state(*visualization_information(specimen))

    plotter.plotter.add_key_event('space', step)

    plotter.open_window()


if __name__ == '__main__':
    main()
