from typing import Callable, Iterable

import numpy as np

import trcdproc.navigate.raw as rawnav
from trcdproc.compute.common import noise
from trcdproc.core import (
    H5File,
    Array,
    subgroups,
    InputChannel,
    PumpStatus,
)
from trcdproc.navigate.common import wavelength_set


def channel_heatmap(file: H5File, func: Callable[[Array, Array], np.float64],
                    channel: InputChannel, pump: PumpStatus) -> Array:
    """Generates a (number of rounds)x(number of wavelengths) array by visiting each wavelength
    group in the file. The value at each position in the array is computed by
    the function provided.

    Note:
        The "func" argument must have the following signature:
            func(time: Array, signal: Array) -> np.float64

    Args:
        file (H5File): The file to generate the heatmap data from
        func (Callable): A function that will be used to compute the heatmap values
        channel (InputChannel): The input channel to use
        pump (PumpStatus): Which dataset (pump or nopump) the data should come from

    Returns:
        A (number of rounds)x(number of wavelengths) array that contains the heatmap pixels
    """
    rounds_root = file['rounds']
    rounds = subgroups(rounds_root)
    unsorted_wavelength_set = wavelength_set(file)
    wavelengths = sorted(list(unsorted_wavelength_set))
    pixels = np.ndarray(shape=(len(rounds), len(wavelengths)), dtype=np.float64)
    for i in range(len(rounds)):
        for j in range(len(wavelengths)):
            group_path = f'{rounds[i]}/{wavelengths[j]}/{pump.value}'
            try:
                group = rounds_root[group_path]
            except KeyError:
                pixels[i, j] = 0
                continue
            time_dset = group['time']
            input_dset = group[f'{channel.value}']
            points = len(group['time'][...])
            # cast float32 data to float64 for computations to avoid rounding errors
            time_data = np.empty(points, dtype=np.float64)
            input_channel_data = np.empty(points, dtype=np.float64)
            time_dset.read_direct(time_data)
            input_dset.read_direct(input_channel_data)
            pixels[i, j] = func(time_data, input_channel_data)
    return pixels


def store_individual_means(src_file: H5File) -> None:
    """Computes the mean of each signal-containing dataset and stores it on the dataset as
    an attribute called 'mean'

    Args:
        src_file (H5File): The file containing the experimental data
    """
    paths = rawnav.all_signal_dataset_paths(src_file)
    for path in paths:
        points = len(src_file[path])
        data = np.empty(points, dtype=np.float64)
        src_file[path].read_direct(data)
        src_file[path].attrs['mean'] = data.mean()
    return


def store_individual_noises(src_file: H5File) -> None:
    """Computes the noise in each non-time dataset and stores it as an attribute called 'noise'

    Args:
        src_file (H5File): The file containing the experimental data
    """
    paths = rawnav.pump_group_paths(src_file)
    for path in paths:
        par_path = path + '/par'
        perp_path = path + '/perp'
        ref_path = path + '/ref'
        points = len(src_file[perp_path])
        for channel in [perp_path, par_path, ref_path]:
            signal_data = np.empty(points, dtype=np.float64)
            src_file[channel].read_direct(signal_data)
            src_file[channel].attrs['noise'] = noise(signal_data)
    return


def store_noise_means(src_file: H5File) -> None:
    """Stores the mean of the noise in each channel as a file-level attribute

    Args:
        src_file (H5File): The file containing the experimental data

    Note:
        The attribute is called '<chan>_noise_mean' where '<chan>' is 'perp', 'par', or 'ref'
    """
    perp_sum = 0
    par_sum = 0
    ref_sum = 0
    counts = 0
    for path in rawnav.pump_group_paths(src_file):
        perp_path = path + '/perp'
        par_path = path + '/par'
        ref_path = path + '/ref'
        perp_sum += src_file[perp_path].attrs['noise']
        par_sum += src_file[par_path].attrs['noise']
        ref_sum += src_file[ref_path].attrs['noise']
        counts += 1
    src_file.attrs['perp_noise_mean'] = perp_sum / counts
    src_file.attrs['par_noise_mean'] = par_sum / counts
    src_file.attrs['ref_noise_mean'] = ref_sum / counts
    return


def store_overall_means(src_file: H5File) -> None:
    """Stores the mean of the mean of each channel as an attribute on the file

    Args:
        src_file (H5File): The file containing the experimental data

    Note:
        The attribute is called '<chan>_mean' where '<chan>' is 'perp', 'par', or 'ref'
    """
    perp_sum = 0
    par_sum = 0
    ref_sum = 0
    counts = 0
    for path in rawnav.pump_group_paths(src_file):
        perp_path = path + '/perp'
        par_path = path + '/par'
        ref_path = path + '/ref'
        perp_sum += src_file[perp_path].attrs['mean']
        par_sum += src_file[par_path].attrs['mean']
        ref_sum += src_file[ref_path].attrs['mean']
        counts += 1
    src_file.attrs['perp_mean'] = perp_sum / counts
    src_file.attrs['par_mean'] = par_sum / counts
    src_file.attrs['ref_mean'] = ref_sum / counts
    return


def store_std_dev_of_means(src_file: H5File) -> None:
    """Stores the standard deviation of the mean of each channel as an attribute on the file

    Args:
        src_file (H5File): The file containing the experimental data

    Note:
        The attribute is called '<chan>_mean_std_dev' where '<chan>' is 'perp', 'par', or 'ref'

    Note:
        Not covered by tests yet
    """
    mean_perp_mean = src_file.attrs['perp_mean']
    mean_par_mean = src_file.attrs['par_mean']
    mean_ref_mean = src_file.attrs['ref_mean']
    perp_sum = 0
    par_sum = 0
    ref_sum = 0
    counts = 0
    for path in rawnav.pump_group_paths(src_file):
        perp_path = path + '/perp'
        par_path = path + '/par'
        ref_path = path + '/ref'
        perp_mean = src_file[perp_path].attrs['mean']
        par_mean = src_file[par_path].attrs['mean']
        ref_mean = src_file[ref_path].attrs['mean']
        perp_sum += (perp_mean - mean_perp_mean) ** 2
        par_sum += (par_mean - mean_par_mean) ** 2
        ref_sum += (ref_mean - mean_ref_mean) ** 2
        counts += 1
    src_file.attrs['perp_mean_std_dev'] = np.sqrt(perp_sum / (counts - 1))
    src_file.attrs['par_mean_std_dev'] = np.sqrt(par_sum / (counts - 1))
    src_file.attrs['ref_mean_std_dev'] = np.sqrt(ref_sum / (counts - 1))
    return


def store_std_dev_of_noises(src_file: H5File) -> None:
    """Stores the standard deviation of the noise in each channel as an attribute on the file

    Args:
        src_file (H5File): The file containing the experimental data

    Note:
        The attribute is called '<chan>_noise_std_dev' where '<chan>' is 'perp', 'par', or 'ref'
    """
    mean_perp_noise = src_file.attrs['perp_noise_mean']
    mean_par_noise = src_file.attrs['par_noise_mean']
    mean_ref_noise = src_file.attrs['ref_noise_mean']
    perp_sum = 0
    par_sum = 0
    ref_sum = 0
    counts = 0
    for path in rawnav.pump_group_paths(src_file):
        perp_path = path + '/perp'
        par_path = path + '/par'
        ref_path = path + '/ref'
        perp_noise = src_file[perp_path].attrs['noise']
        par_noise = src_file[par_path].attrs['noise']
        ref_noise = src_file[ref_path].attrs['noise']
        perp_sum += (perp_noise - mean_perp_noise) ** 2
        par_sum += (par_noise - mean_par_noise) ** 2
        ref_sum += (ref_noise - mean_ref_noise) ** 2
        counts += 1
    src_file.attrs['perp_noise_std_dev'] = np.sqrt(perp_sum / (counts - 1))
    src_file.attrs['par_noise_std_dev'] = np.sqrt(par_sum / (counts - 1))
    src_file.attrs['ref_noise_std_dev'] = np.sqrt(ref_sum / (counts - 1))
    return


def store_all_noise_attributes(src_file: H5File) -> None:
    """Stores the noise attributes for each of the signals

    The noise attributes stored on each signal are used for filtering out bad data. The following
    attributes are stored on individual signals:
     - `noise`
     - `mean`
    The following attributes are stored as attributes at the top level of the file:
     - `<sig>_mean`
     - `<sig>_noise_mean`
     - `<sig>_mean_std_dev`
     - `<sig>_noise_std_dev`
    where `<sig>` is one of `perp`, `par`, or `ref`.

    Args:
        src_file (H5File): The file containing the experiment data
    """
    store_individual_means(src_file)
    store_individual_noises(src_file)
    store_overall_means(src_file)
    store_noise_means(src_file)
    store_std_dev_of_means(src_file)
    store_std_dev_of_noises(src_file)
    return


def min_noise(src_file: H5File, paths: Iterable[str]) -> np.float64:
    """Returns the minimum noise found in the file

    Args:
        src_file (H5File): The file containing the experimental data
        paths (Generator[str]): An iterator over the paths of each signal dataset

    Note:
        Not covered by tests yet

    Returns:
        The minimum noise found in the file
    """
    smallest_noise = 1000.0
    for path in paths:
        if smallest_noise > src_file[path].attrs['noise']:
            smallest_noise = src_file[path].attrs['noise']
    return smallest_noise
