import numpy as np


def shape_fill_none(array: np.array, fill_upto: int):
    """
    Does the same as np.shape but fills non-existing dimensions with None
    :param array: the array which shape should be returned
    :param fill_upto: maximum number of dimensions length of the returned tuple when the array has less dimensions
    """
    shape = np.shape(array)
    if len(shape) < fill_upto:
        shape = shape + (None,) * (fill_upto - len(shape))
    return shape