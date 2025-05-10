import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvista as pv
import keyboard

from neuron.data_structure import Data
from neuron.genome import init_genome, Genome, InitialNeuronConfiguration
from neuron.specimen import Specimen


def draw_specimen2(specimen: Specimen):
    scale = 60
    neurons = specimen.neurons[torch.tensor(specimen.living_neuron_indices, dtype=torch.int)]
    background = torch.zeros((scale * 10, scale * 10), dtype=torch.float)
    positions = ((neurons[:, Data.POSITION.value] - 15) * scale).to(dtype=torch.int)
    positions = positions[(positions[:, 1] > 0) & (positions[:, 1] < scale * 10 - 1)]
    positions = positions[(positions[:, 0] > 0) & (positions[:, 0] < scale * 10 - 1)]
    for yo, xo in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        background[positions[:, 1] + yo, positions[:, 0] + xo] = 1.0
    # for neuron in neurons:
    #     x, y, z = (neuron[Data.POSITION.value] * scale).to(dtype=torch.int)
    #     background[y - 2:y + 2, x - 2:x + 2] = 1.0

    image = background.unsqueeze(2).repeat((1, 1, 3))
    cv2.imshow('image', image.numpy())
    # plt.imshow(image)
    # plt.show()


class Plotter:
    def __init__(self):
        self.plotter = pv.Plotter(window_size=[800, 600])
        self.points_actor: Optional[pv.Actor] = None
        self.lines_actor: Optional[pv.Actor] = None

    def open_window(self):
        self.plotter.camera_position = 'iso'
        # self.plotter.add_axes()
        # self.plotter.show_grid(color='gray')  # Culling=False makes grid always visible
        self.plotter.show(auto_close=False)
        # self.plotter.enable_depth_peeling()

    def close_window(self):
        pass

    def set_state(
        self,
        coordinates: np.ndarray,
        point_colors: np.ndarray,
        # connection_strengths: np.ndarray,
        connection_colors: np.ndarray
    ):
        point_cloud_mesh = pv.PolyData(coordinates)
        point_cloud_mesh.point_data['neuron_colors'] = point_colors
        point_cloud_mesh.active_scalars_name = 'neuron_colors'  # Tell plotter which scalars to use
        lines_mesh = pv.PolyData(coordinates, lines=_line_segment_indices(coordinates.shape[0]))
        lines_mesh['line_colors'] = connection_colors
        # lines_mesh.active_scalars_name = 'line_colors'  # Tell plotter which scalars to use
        if self.points_actor is None:
            self.points_actor = self.plotter.add_mesh(
                point_cloud_mesh,
                style='points',  # 'points' or 'spheres'
                render_points_as_spheres=True,  # Looks better
                point_size=10,
                scalars='neuron_colors',
                name='dynamic_points'
            )
            self.lines_actor = self.plotter.add_mesh(
                lines_mesh,
                scalars='line_colors',
                line_width=2,
                name='dynamic_lines',
                rgba=True
            )
        else:
            self.points_actor.mapper.dataset = point_cloud_mesh
            # self.points_actor.mapper.scalar_visibility = True  # Ensure scalars are used for coloring
            self.points_actor.mapper.SetScalarModeToUsePointData()  # Use point data for coloring

            self.lines_actor.mapper.dataset = lines_mesh
            # self.lines_actor.mapper.SetScalarModeToUsePointData()  # Use point data for coloring

        self.plotter.render()


def _line_segment_indices(m: int) -> np.ndarray:
    temp = np.expand_dims(np.arange(m, dtype=int), 1).repeat(m, axis=1)
    i = temp.flatten()
    j = temp.T.flatten()
    segments = np.stack([np.full_like(i, 2), i, j]).T
    lines = segments.flatten()
    return lines


def draw_specimen(specimen: Specimen):
    plotter = pv.Plotter(window_size=[800, 600])

    num_current_points = np.random.randint(5, 10 + 1)
    current_points = np.random.rand(num_current_points, 3) * 5 - 2.5  # Points in a cube
    current_points[:, 2] *= np.sin(i * 0.1)  # Animate Z coordinates

    # Random colors (RGBA, 0-255)
    current_colors = (np.random.rand(num_current_points, 3) * 255).astype(np.uint8)
    point_cloud_mesh = pv.PolyData(current_points)
    point_cloud_mesh.point_data['frame_colors'] = current_colors
    point_cloud_mesh.active_scalars_name = 'frame_colors'  # Tell plotter which scalars to use

    # initial_points = np.empty((0, 3), dtype=float)
    # initial_point_cloud = pv.PolyData(initial_points)
    # initial_lines_mesh = pv.PolyData(np.array([[0, 1]]))
    lines_mesh = pv.PolyData(current_points, lines=np.array([2, 0, 1]))
    points_actor = plotter.add_mesh(point_cloud_mesh,
                                    style='points',  # 'points' or 'spheres'
                                    render_points_as_spheres=True,  # Looks better
                                    point_size=10,
                                    color='lightblue',  # Default color if no scalars
                                    name='dynamic_points')

    lines_actor = plotter.add_mesh(lines_mesh,
                                   color='grey',
                                   line_width=2,
                                   name='dynamic_lines')

    plotter.camera_position = 'iso'
    plotter.add_axes()
    plotter.show_grid(color='gray')  # Culling=False makes grid always visible
    plotter.show(interactive_update=True, auto_close=False)
    print("Starting animation loop. Close the PyVista window to stop.")
    # 1. Generate new data for this frame
    num_current_points = np.random.randint(5, 10 + 1)
    current_points = np.random.rand(num_current_points, 3) * 5 - 2.5  # Points in a cube
    current_points[:, 2] *= np.sin(i * 0.1)  # Animate Z coordinates

    # Random colors (RGBA, 0-255)
    current_colors = (np.random.rand(num_current_points, 3) * 255).astype(np.uint8)
    point_cloud_mesh = pv.PolyData(current_points)
    point_cloud_mesh.point_data['frame_colors'] = current_colors
    point_cloud_mesh.active_scalars_name = 'frame_colors'  # Tell plotter which scalars to use

    # Update the actor's mesh
    # This replaces the underlying geometry and scalar data of the actor
    points_actor.mapper.dataset = point_cloud_mesh
    points_actor.mapper.scalar_visibility = True  # Ensure scalars are used for coloring
    points_actor.mapper.SetScalarModeToUsePointData()  # Use point data for coloring

    # Random connections (ensure indices are valid)
    num_connections = np.random.randint(0, num_current_points * 2)
    connections_list = []
    if num_current_points > 1:
        for _ in range(num_connections):
            c = np.random.choice(num_current_points, 2, replace=False)
            connections_list.append((c[0], c[1]))

    # --- Update Connections ---
    if connections_list:
        lines_cell_array = []
        for conn_start, conn_end in connections_list:
            lines_cell_array.extend([2, conn_start, conn_end])
        # Connections use the same point coordinates as the point_cloud_mesh
        lines_mesh = pv.PolyData(current_points, lines=np.array(lines_cell_array))
        lines_actor.mapper.dataset = lines_mesh
        if lines_actor not in plotter.actors.values():  # Add if removed
            plotter.add_actor(lines_actor, name='dynamic_lines')

    # 3. Render the scene
    plotter.render()
    # plotter.update() # Alternative, sometimes preferred

    # # 4. Frame rate control
    # elapsed_time = time.perf_counter() - start_time
    # sleep_time = FRAME_DELAY - elapsed_time
    # if sleep_time > 0:
    #     time.sleep(sleep_time)

    plotter.show(interactive=True)
    # while plotter.iren and plotter.iren.initialized:
    #     time.sleep(0.2)
    print("Plotter window closed, exiting animation.")
    # if i % 10 == 0:  # Print FPS occasionally
    #     actual_fps = 1.0 / (elapsed_time if elapsed_time > 0 else 0.001)
    #     print(f"Frame {i}/{N_FRAMES}, Approx FPS: {actual_fps:.1f}")


def create_specimen(genome: Genome) -> Specimen:
    specimen = Specimen(genome, [
        InitialNeuronConfiguration(None, torch.tensor([0.0, 0.0, 0.0])),
        InitialNeuronConfiguration(None, torch.tensor([2.0, 0.0, 0.0]))
    ])
    return specimen
genome = torch.load('../../checkpoints/2025-05-09/19-36-11/generation_39_survivors.pt', weights_only=False)[0]
# genome = init_genome()
# specimen = Specimen(genome)
# specimen.add_neurons(
#     torch.Tensor([[20, 20, 0], [22, 20, 0]]),
#     genome.pluripotent_latent_state.unsqueeze(0).repeat((2, 1))
# )
specimen = create_specimen(genome)

plotter = Plotter()
start = time.time()

def step():
    specimen.step()
    # neurons = specimen.neurons[specimen.living_neuron_indices]
    coordinates = specimen.log.neuron_positions.detach().cpu().numpy() / 10

    connectivity = specimen.log.connectivity.detach().cpu().numpy().flatten()
    connection_opacity = np.clip(connectivity.flatten() * 10, 0.0, 1.0)
    connection_opacity[np.argsort(connectivity)[:-connection_opacity.shape[0] // 10]] = 0
    connection_color = np.zeros((connectivity.shape[0], 4))
    connection_color[:] = np.array([0.6, 0.6, 0.85, 0.0])
    connection_color[:, -1] = connection_opacity

    plotter.set_state(
        coordinates,
        (np.random.rand(coordinates.shape[0], 3) * 255).astype(np.uint8),
        connection_color
    )
# quit = threading.Event()
# keyboard.add_hotkey(' ', step)
# keyboard.add_hotkey('q', lambda: quit.set())


# while not quit.is_set():
#     time.sleep(0.1)
#
# plotter.close_window()
quit = threading.Event()
# plotter.plotter.add_key_event('q', lambda: quit.set())
plotter.plotter.add_key_event('space', step)
plotter.open_window()
# for i in range(350):
#     # print('Stepping')
#     specimen.step()
#     # print(len(specimen.living_neuron_indices))
#     # print(i)
#     # print('Stepped', time.time() - start)
#     # draw_specimen(specimen)
#     neurons = specimen.neurons[specimen.living_neuron_indices]
#     coordinates = neurons[:, Data.POSITION.value].detach().cpu().numpy()
#     print(coordinates.shape)
#     plotter.set_state(coordinates, (np.random.rand(neurons.size(0), 3) * 255).astype(np.uint8))
#     # key = cv2.waitKey(0)
#     key = plotter.plotter.key_press_event()
#     # while key.event_type != 'down':
#     #     key = keyboard.read_event()
#     if quit.is_set():
#         break
# print(time.time() - start)
# while not quit.is_set():
#     time.sleep(0.1)
plotter.close_window()
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
