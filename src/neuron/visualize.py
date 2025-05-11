from typing import Optional

import numpy as np
import pyvista as pv
import torch

from neuron.genome import Genome, InitialNeuronConfiguration
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
        point_cloud_mesh.point_data['neuron_colors'] = point_colors
        point_cloud_mesh.active_scalars_name = 'neuron_colors'
        lines_mesh = pv.PolyData(coordinates, lines=_line_segment_indices(coordinates.shape[0]))
        lines_mesh['line_colors'] = connection_colors
        if self.points_actor is None:
            self.points_actor = self.plotter.add_mesh(
                point_cloud_mesh,
                style='points',  # 'points' or 'spheres'
                render_points_as_spheres=True,  # Looks better
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
            self.points_actor.mapper.scalar_visibility = True
            self.points_actor.mapper.SetScalarModeToUsePointData()
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
        InitialNeuronConfiguration(None, torch.tensor([2.0, 0.0, 0.0]))
    ])
    return specimen


def main():
    genome = torch.load('../../checkpoints/2025-05-09/19-36-11/generation_39_survivors.pt', weights_only=False)[0]
    # genome = init_genome()
    # specimen = Specimen(genome)
    # specimen.add_neurons(
    #     torch.Tensor([[20, 20, 0], [22, 20, 0]]),
    #     genome.pluripotent_latent_state.unsqueeze(0).repeat((2, 1))
    # )
    specimen = create_specimen(genome)

    plotter = Plotter()

    def step():
        specimen.stimulate_input_neurons({0: 0.2, 1: 0.1})
        specimen.step()
        # neurons = specimen.neurons[specimen.living_neuron_indices]
        coordinates = specimen.log.neuron_positions.detach().cpu().numpy() / 10
        activations = specimen.log.activations.detach().cpu().numpy()
        neuron_color = np.zeros((coordinates.shape[0], 3))
        neuron_color[:] = np.array([0.1, 0.1, 1.0])
        # print(activations.shape, neuron_color.shape)
        neuron_color[activations] = np.array([1.0, 1.0, 0.1])

        activations_matrix = np.zeros((coordinates.shape[0], coordinates.shape[0]), dtype=np.bool)
        activations_matrix[activations] = True

        connectivity = specimen.log.connectivity.detach().cpu().numpy().flatten()
        connection_opacity = np.clip(connectivity.flatten() * 2, 0.0, 1.0)
        # connection_opacity[np.argsort(connectivity)[:-connection_opacity.shape[0] // 10]] = 0  # cull bottom 90% connections
        connection_color = np.zeros((connectivity.shape[0], 4))
        connection_color[:] = np.array([0.6, 0.6, 0.85, 0.0])
        # print(activations.shape, connection_color.shape)
        connection_color[:, -1] = connection_opacity
        connection_color[activations_matrix.flatten()] = np.array([1.0, 1.0, 0.1, 1.0])

        plotter.set_state(
            coordinates,
            neuron_color,
            connection_color
        )

    plotter.plotter.add_key_event('space', step)

    plotter.open_window()


if __name__ == '__main__':
    main()
