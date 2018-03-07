import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
from logzero import logger

import trcdproc.compute.absorp as abcomp
import trcdproc.compute.raw as rawcomp
import trcdproc.filter.raw as rawfilter
import trcdproc.reorganize.raw as reorgraw
from trcdproc.core import Group, H5File, subgroups


def restructure_and_merge(filenames: List[str], joined_name: str='joined.h5') -> None:
    """Restructures and merges the data located at the filenames provided

    Args:
        filenames (List[str]): A list of filenames to restructure and merge
        joined_name (str): The name of the file in which to store the merged data

    Warning:
        This function can use a large amount of disk space (at most ~2x the size of the original
        data) because it has to copy the original data into temporary files, then merge the
        temporary files. The temporary files are removed at the end.
    """
    counter = 0
    restructured_filenames = []
    for filename in filenames:  # restructure the original data into new files
        src_file = h5py.File(filename, 'r', libver='latest')
        restructured_name = f'restructured{counter:0>3d}.h5'
        restructured_filenames.append(restructured_name)
        dest_file = h5py.File(restructured_name, 'w', libver='latest')
        reorgraw.reorganize_rounds(src_file, dest_file)
        src_file.close()
        dest_file.close()
        counter += 1
    reorgraw.combine_files(restructured_filenames, joined_name)
    for filename in restructured_filenames:  # remove the temporary files
        os.remove(filename)
    return


def conservatively_mark_bad_data(experiment_data: H5File, noise_floor: float, filter_size: float) -> None:
    """Uses the supplied filter parameters to mark the data that should not be included in later
    computations or averaging.

    Data is marked as good or bad by creating an attribute `isbad` on wavelength groups and
    setting it to `True` or `False`.

    There are three filters that are applied to the data:
    - The first filter simply marks any wavelength group as bad if it does not contain both a `pump`
      group and a `nopump` group, as both are required in order to be able to calculate the change
      in absorption. There are no settings for this filter.
    - The second filter looks for datasets with unreasonably low noise. This is meant to catch any
      datasets for which the values in the dataset are approximately constant. It's not currently
      known why this happens, but it happens nonetheless. There is one setting for this filter: the
      minimum allowable noise. This value is set by the `noise_floor` parameter, and must be chosen
      through trial and error to prevent throwing away good data.
    - The third filter catches datasets with unreasonably high noise. Datasets with high noise are
      selected by looking at the difference between the noise of a particular dataset and the mean
      of the noise in the corresponding channel. The setting for this filter is the maximum
      difference between the individual and mean noises as determined by the `filter_size`
      parameter. A dataset is marked bad if the difference between its noise and the mean noise in
      that channel is greater than the product of `filter_size` and the standard deviation of the
      noise in that channel. Put another way, a dataset is marked as bad if it is more than
      `filter_size` standard deviations away from the mean noise in that channel.

    Args:
        experiment_data (H5File): An HDF5 file that contains restructured experiment data
        noise_floor (float): The minimum allowable noise in a dataset
        filter_size (float): The parameter for tuning the selectivity of the high noise filter
    """
    rawcomp.store_all_noise_attributes(experiment_data)
    faulty = rawfilter.filter_faulty_groups(experiment_data)
    too_little_noise = rawfilter.filter_noise_too_low(experiment_data, noise_floor)
    too_much_noise = rawfilter.filter_noise_too_high(experiment_data, filter_size)
    all_filtered_paths = faulty + too_little_noise + too_much_noise
    rawfilter.mark_bad_wavelength_groups(experiment_data, all_filtered_paths)
    return


def selectively_mark_bad_data(experiment_data: H5File, noise_floor: float, filter_size: float) -> None:  # noqa
    """Uses the supplied filter parameters to mark the data that should not be included in later
    computations or averaging.

    Data is marked as good or bad by creating an attribute `isbad` on wavelength groups and
    setting it to `True` or `False`.

    There are three filters that are applied to the data:
    - The first filter simply marks any wavelength group as bad if it does not contain both a `pump`
      group and a `nopump` group, as both are required in order to be able to calculate the change
      in absorption. There are no settings for this filter.
    - The second filter looks for datasets with unreasonably low noise. This is meant to catch any
      datasets for which the values in the dataset are approximately constant. It's not currently
      known why this happens, but it happens nonetheless. There is one setting for this filter: the
      minimum allowable noise. This value is set by the `noise_floor` parameter, and must be chosen
      through trial and error to prevent throwing away good data.
    - The third filter catches datasets with unreasonably high noise. Datasets with high noise are
      selected by looking at the difference between the noise of a particular dataset and the mean
      of the noise in the corresponding channel. The setting for this filter is the maximum
      difference between the individual and mean noises as determined by the `filter_size`
      parameter. A dataset is marked bad if the difference between its noise and the mean noise in
      that channel is greater than the product of `filter_size` and the standard deviation of the
      noise in that channel. Put another way, a dataset is marked as bad if it is more than
      `filter_size` standard deviations away from the mean noise in that channel.

    Args:
        experiment_data (H5File): An HDF5 file that contains restructured experiment data
        noise_floor (float): The minimum allowable noise in a dataset
        filter_size (float): The parameter for tuning the selectivity of the high noise filter
    """
    logger.info(f'Computing noise attributes for {experiment_data.filename}')
    rawcomp.store_all_noise_attributes(experiment_data)
    logger.info(f'Applying filters for {experiment_data.filename}')
    faulty = rawfilter.filter_faulty_groups(experiment_data)
    too_little_noise = rawfilter.filter_noise_too_low(experiment_data, noise_floor)
    too_much_noise = rawfilter.filter_noise_too_high(experiment_data, filter_size)
    all_filtered_paths = faulty + too_little_noise + too_much_noise
    logger.info(f'Marking filtered signals in {experiment_data.filename}')
    rawfilter.mark_bad_signals(experiment_data, all_filtered_paths)
    return


def compute_averaged_delta_a(src_filename: str, dest_filename: str, noise_floor: float,
                             filter_size: float, include_cd: bool=False) -> None:
    """Computes the change in absorption from the rounds contained in the source file, then
    averages the data together after the baselines have all been adjusted to zero

    Args:
        src_filename (str): The filename from which to read the data
        dest_filename (str): The filename in which to store the results
        include_cd (bool): Whether to include circular dichroism in the results
    """
    # Mark the noisy, incomplete, or otherwise faulty data
    src_file = h5py.File(src_filename, 'r+', libver='latest')
    conservatively_mark_bad_data(src_file, noise_floor=noise_floor, filter_size=filter_size)
    # Compute the change in absorption
    delta_a_file = h5py.File('delta-a.h5', 'w', libver='latest')
    abcomp.compute_only_good_delta_a(src_file, delta_a_file, include_cd=include_cd)
    src_file.close()
    # Adjust the baseline of each piece of data
    results_file = h5py.File(dest_filename, 'w', libver='latest')
    abcomp.adjust_baseline(delta_a_file, results_file, include_cd=include_cd)
    delta_a_file.close()
    os.remove('delta-a.h5')
    # Average the results
    abcomp.average_rounds(results_file, include_cd=include_cd)
    results_file.close()
    return


def compute_selectively_averaged_delta_a(src_filename: str, dest_filename: str, noise_floor: float,
                                         filter_size: float, include_cd: bool=False) -> None:
    """Computes the change in absorption from the rounds contained in the source file, then
    averages the data together after the baselines have all been adjusted to zero

    Args:
        src_filename (str): The filename from which to read the data
        dest_filename (str): The filename in which to store the results
        include_cd (bool): Whether to include circular dichroism in the results
    """
    # Mark the noisy, incomplete, or otherwise faulty data
    logger.info(f'Marking bad data in {src_filename}')
    src_file = h5py.File(src_filename, 'r+', libver='latest')
    selectively_mark_bad_data(src_file, noise_floor=noise_floor, filter_size=filter_size)
    # Compute the change in absorption
    logger.info(f'Computing dA from {src_filename}')
    delta_a_file = h5py.File('delta-a.h5', 'w', libver='latest')
    abcomp.compute_delta_a_selectively(src_file, delta_a_file, include_cd=include_cd)
    src_file.close()
    # Adjust the baseline of each piece of data
    logger.info(f'Adjusting baseline from delta-a.h5 into {dest_filename}')
    results_file = h5py.File(dest_filename, 'w', libver='latest')
    abcomp.adjust_baseline(delta_a_file, results_file, include_cd=include_cd)
    delta_a_file.close()
    os.remove('delta-a.h5')
    # Average the results
    logger.info(f'Averaging data in {dest_filename}')
    abcomp.average_selectively(results_file, include_cd=include_cd)
    results_file.close()
    logger.info(f'Done with {dest_filename}')
    return


def export_txt_data(data_group: Group, sample_name: str, folder_name: str, include_cd: bool=False) -> None: # noqa
    """Point this function at a group containing wavelength groups to export the data
    as CSV files.

    A folder with the specified name is created in the working directory, and under that
    folder a folder is created for each signal ('perp', 'par', and optionally 'cd'). The
    folder hierarchy is thus:
        <folder_name>
        |---perp
        |---par
        |---cd

    The data corresponding to each channel is placed in its respective folder. The
    filenames are formatted as follows:
        <sample_name>-<wavelength>-<signal>.txt

    Note that although the data is saved in CSV format, the file extension is `.txt`. This
    is for compatibility with Spectra Solve, which expects comma-delimitted data to have
    a `.txt` extension (for some reason).

    Args:
        data_group (Group): The group containing the data to export
        sample_name (str): A sample name to include in the filename of each piece of data
        folder_name (str): The name of the folder in which to store the exported data
        include_cd (bool): Indicates whether to export CD data
    """
    data_dir = Path.cwd() / folder_name
    if data_dir.exists() and data_dir.is_dir():
        logger.error(f'Directory {data_dir} already exists, please choose another name')
        return
    data_dir.mkdir()
    signals = ['perp', 'par']
    if include_cd:
        signals.append('cd')
    signal_dirs = {
        'perp': data_dir / 'perp',
        'par': data_dir / 'par',
    }
    if include_cd:
        signal_dirs['cd'] = data_dir / 'cd'
    for sig_dir in signal_dirs.values():
        sig_dir.mkdir()
    wavelengths = subgroups(data_group)
    for wav in wavelengths:
        wav_group = data_group[wav]
        time_data = wav_group['time'][...]
        for sig in signals:
            sig_data = wav_group[sig][...]
            two_column_data = np.column_stack((time_data, sig_data))
            file_path = signal_dirs[sig] / f'{sample_name}-{wav}-{sig}.txt'
            logger.info(f'Saving dataset {wav_group[sig].name} to file {file_path}')
            with file_path.open(mode='wb') as file:
                np.savetxt(file, two_column_data, delimiter=',')
    return
