from typing import Union, Tuple
# data formatting strategy for efficiency and compactness:
# 1. slot all neuron data into a single 1d tensor, and then stack each neuron tensor.
#    the position of a neuron tensor is its ID except for position 0, which is a dummy
#    for computational purposes
# 2. Genome can just live in a class
from enum import Enum, IntEnum

_HORMONE_DIM = 10
_LATENT_DIM = 24
MAX_CONNECTIONS = 8


class _Properties(IntEnum):
    ACTIVATION_THRESHOLD = 0
    SIGNAL_STRENGTH = 1
    HORMONE_EMISSION = 2
    HORMONE_RANGE = 3

    ACTIVATION_WARMUP = 4
    CELL_DAMAGE = 5
    MITOSIS_STAGE = 6

    LATENT_STATE = 7
    TOTAL_RECEPTIVITY = 8
    TOTAL_EMISSIVITY = 9
    ACTIVATION_PROGRESS = 10
    HORMONE_INFLUENCE = 11

    POSITION = 12
    INPUT_INDICES = 13
    INPUT_CONNECTIVITY = 14
    OUTPUT_INDICES = 15
    OUTPUT_CONNECTIVITY = 16


_DATA_SIZES = {
    _Properties.ACTIVATION_THRESHOLD: 1,
    _Properties.SIGNAL_STRENGTH: 1,
    _Properties.HORMONE_EMISSION: _HORMONE_DIM,
    _Properties.HORMONE_RANGE: 1,

    _Properties.ACTIVATION_WARMUP: 1,
    _Properties.CELL_DAMAGE: 1,
    _Properties.MITOSIS_STAGE: 1,

    _Properties.LATENT_STATE: _LATENT_DIM,
    _Properties.TOTAL_RECEPTIVITY: 1,
    _Properties.TOTAL_EMISSIVITY: 1,
    _Properties.ACTIVATION_PROGRESS: 1,
    _Properties.HORMONE_INFLUENCE: _HORMONE_DIM,

    _Properties.POSITION: 3,
    _Properties.INPUT_INDICES: MAX_CONNECTIONS,
    _Properties.INPUT_CONNECTIVITY: MAX_CONNECTIONS,
    _Properties.OUTPUT_INDICES: MAX_CONNECTIONS,
    _Properties.OUTPUT_CONNECTIVITY: MAX_CONNECTIONS,
}


def _get_indexing(property_name: _Properties) -> Union[int, slice]:
    property_names, property_sizes = list(_DATA_SIZES.keys()), list(_DATA_SIZES.values())
    property_index = property_names.index(property_name)
    property_size = property_sizes[property_index]
    data_format_start_index = sum(property_sizes[:property_index], 0)
    return data_format_start_index if property_size == 1 else slice(data_format_start_index,
                                                                    data_format_start_index + property_size)


def _get_block_segment(start_property: _Properties, end_property: _Properties,
                       offset_from: _Properties = None) -> slice:
    property_names, property_sizes = list(_DATA_SIZES.keys()), list(_DATA_SIZES.values())
    start_index = property_names.index(start_property)
    end_index = property_names.index(end_property)
    data_format_start_index = sum(property_sizes[:start_index], 0)
    data_format_end_index = sum(property_sizes[:end_index + 1], 0)
    if offset_from is not None:
        offset = sum(property_sizes[:property_names.index(offset_from)], 0)
        data_format_start_index -= offset
        data_format_end_index -= offset
    return slice(data_format_start_index, data_format_end_index)


# NOTE: when editing the data format, be careful about reordering properties,
#       as some definitions rely on subsets of properties being in a continuous chunk
class Data(Enum):
    # State-derived Parameters -----------
    # - activation parameters
    ACTIVATION_THRESHOLD = _get_indexing(_Properties.ACTIVATION_THRESHOLD)
    SIGNAL_STRENGTH = _get_indexing(_Properties.SIGNAL_STRENGTH)
    # - hormone emission
    HORMONE_EMISSION = _get_indexing(_Properties.HORMONE_EMISSION)
    HORMONE_RANGE = _get_indexing(_Properties.HORMONE_EMISSION)

    # State Parameters -------------------
    # - incremented/decremented parameters
    ACTIVATION_WARMUP = _get_indexing(_Properties.ACTIVATION_WARMUP)
    CELL_DAMAGE = _get_indexing(_Properties.CELL_DAMAGE)
    MITOSIS_STAGE = _get_indexing(_Properties.MITOSIS_STAGE)

    # - direct parameters
    LATENT_STATE = _get_indexing(_Properties.LATENT_STATE)
    TOTAL_RECEPTIVITY = _get_indexing(_Properties.TOTAL_RECEPTIVITY)
    TOTAL_EMISSIVITY = _get_indexing(_Properties.TOTAL_EMISSIVITY)
    ACTIVATION_PROGRESS = _get_indexing(_Properties.ACTIVATION_PROGRESS)
    HORMONE_INFLUENCE = _get_indexing(_Properties.HORMONE_INFLUENCE)

    # Hidden State Parameters -------------
    POSITION = _get_indexing(_Properties.POSITION)
    INPUT_INDICES = _get_indexing(_Properties.INPUT_INDICES)
    INPUT_CONNECTIVITY = _get_indexing(_Properties.INPUT_CONNECTIVITY)
    OUTPUT_INDICES = _get_indexing(_Properties.OUTPUT_INDICES)
    OUTPUT_CONNECTIVITY = _get_indexing(_Properties.OUTPUT_CONNECTIVITY)

    # Shortcuts
    # - shortcut slices to subsections of above parameters
    DERIVED_PARAMETERS = _get_block_segment(_Properties.ACTIVATION_THRESHOLD, _Properties.HORMONE_RANGE)
    STATE = _get_block_segment(_Properties.ACTIVATION_WARMUP, _Properties.HORMONE_INFLUENCE)
    INCREMENTED_PARAMETERS = _get_block_segment(_Properties.ACTIVATION_WARMUP, _Properties.MITOSIS_STAGE)

    TRANSFORM_INCREMENTED = _get_block_segment(
        _Properties.ACTIVATION_WARMUP, _Properties.MITOSIS_STAGE, offset_from=_Properties.ACTIVATION_WARMUP)
    TRANSFORM_LATENT = _get_block_segment(
        _Properties.LATENT_STATE, _Properties.LATENT_STATE, offset_from=_Properties.ACTIVATION_WARMUP)
    MITOSIS_PARENT_LATENT = slice(0, _LATENT_DIM)
    MITOSIS_CHILD_LATENT = slice(_LATENT_DIM, _LATENT_DIM * 2)
    MITOSIS_SPLIT_POSITION = slice(_LATENT_DIM * 2, _LATENT_DIM * 2 + 3)


NEURON_DATA_DIM = sum(_DATA_SIZES.values())
NEURON_DIAMETER = 1

