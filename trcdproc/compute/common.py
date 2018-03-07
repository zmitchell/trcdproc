from typing import Callable

import numpy as np
from scipy.interpolate import UnivariateSpline

from trcdproc.core import subgroups, copy_all_attributes, H5File, Group, Array


def map_rounds_datasets(old_file: H5File, new_file: H5File,
                        func: Callable[[Group, Group], None]) -> None:
    """Visits each wavelength group in each round and applies a function mapping datasets in the
    old group to datasets in the new group

    Note:
        The mapping function must have the following signature:
            func(old_group: Group, new_group: Group) -> None

    Note:
        Not covered by tests yet

    Args:
        old_file (H5File): The file containing the original data
        new_file (H5File): The file to store the mapped data into
        func (Callable): The function that will be used to map data from the old file to the new file
    """
    old_rounds_root = old_file['rounds']
    new_rounds_root = new_file['rounds']
    for rnd_name in subgroups(old_rounds_root):
        for wav_name in subgroups(old_rounds_root[rnd_name]):
            wav_path = f'{rnd_name}/{wav_name}'
            old_group = old_rounds_root[wav_path]
            new_group = new_rounds_root[wav_path]
            func(old_group, new_group)
    copy_all_attributes(old_file, new_file)
    return


def noise(signal: Array) -> np.float64:
    """Determines the noise in the provided data, even if it isn't centered around zero

    A cubic spline is fit to the data to model the overall shape, then the spline is
    subtracted from the data to pick out deviations from the local average. The noise
    is the standard deviation of the difference between the raw data and the spline.

    Args:
        signal (Array): The signal whose noise will be computed

    Returns:
        The noise of the signal as a 64-bit float
    """
    points = len(signal)
    x_data = np.arange(0, points, 1, dtype=np.float64)
    spline = UnivariateSpline(x_data, signal)
    smoothed = spline(x_data)
    deviations = signal - smoothed
    return deviations.std()
