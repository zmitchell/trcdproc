from typing import List

import numpy as np
from logzero import logger

import trcdproc.navigate.raw as rawnav
from trcdproc.core import H5File, subgroups
from trcdproc.navigate.common import dataset_paths_below


def filter_faulty_groups(src_file: H5File) -> List[str]:
    """Returns the list of signal dataset paths contained in faulty groups

    Args:
        src_file (H5File): The file containing the experimental data

    Returns:
        The paths to all of the signal datasets in faulty groups
    """
    wav_paths = rawnav.wavelengths_under_rounds_paths(src_file)
    faulty_paths = []
    for wav_path in wav_paths:
        pump_groups = subgroups(src_file[wav_path])
        if 'faulty' not in pump_groups[0]:
            continue
        for p_name in pump_groups:
            pump_path = f'{wav_path}/{p_name}'
            paths_below = [p for p in dataset_paths_below(src_file, pump_path) if 'time' not in p]
            faulty_paths += paths_below
    return faulty_paths


def filter_noise_too_high(src_file: H5File, filter_size: float) -> List[str]:
    """Returns the list of dataset_paths that contain noise larger than some multiple of the
    standard deviation of that channel's noise

    Args:
        src_file (H5File): The file containing the experimental data
        filter_size (float): Datasets with noise larger than (filter_size)*(noise std. dev.)
            will be considered bad

    Returns:
        The list of paths to the bad datasets

    Note:
        If the reference channel is considered bad, then all of the datasets in that group
        are considered bad. During the calculation of dA each channel is divided by reference,
        so a bad reference channel will spoil the rest of the data.
    """
    pump_paths = rawnav.pump_group_paths(src_file)
    mean_perp_noise = src_file.attrs['perp_noise_mean']
    mean_par_noise = src_file.attrs['par_noise_mean']
    mean_ref_noise = src_file.attrs['ref_noise_mean']
    perp_cutoff = filter_size * src_file.attrs['perp_noise_std_dev']
    par_cutoff = filter_size * src_file.attrs['par_noise_std_dev']
    ref_cutoff = filter_size * src_file.attrs['ref_noise_std_dev']
    bad_paths = []
    for pump_path in pump_paths:
        perp_path = f'{pump_path}/perp'
        par_path = f'{pump_path}/par'
        ref_path = f'{pump_path}/ref'
        ref_noise = src_file[ref_path].attrs['noise']
        if np.abs(ref_noise - mean_ref_noise) > ref_cutoff:
            # bad_paths.append(perp_path)
            # bad_paths.append(par_path)
            bad_paths.append(ref_path)
            logger.debug(f'Noise too high: {src_file.filename}::{ref_path}')
            # continue
        perp_noise = src_file[perp_path].attrs['noise']
        par_noise = src_file[par_path].attrs['noise']
        if np.abs(perp_noise - mean_perp_noise) > perp_cutoff:
            bad_paths.append(perp_path)
            logger.debug(f'Noise too high: {src_file.filename}::{perp_path}')
        if np.abs(par_noise - mean_par_noise) > par_cutoff:
            bad_paths.append(par_path)
            logger.debug(f'Noise too high: {src_file.filename}::{par_path}')
    return bad_paths


def filter_noise_too_low(src_file: H5File, noise_floor: float) -> List[str]:
    """Returns the list of dataset_paths that contain Noise too small to correspond
    to a real signal

    Args:
        src_file (H5File): The file containing the experimental data
        noise_floor (np.float64): Datasets with noise lower than 'noise_floor' are thrown out

    Returns:
        The list of paths to the bad datasets

    Note:
        If the reference channel is considered bad, then all of the datasets in that group
        are considered bad. During the calculation of dA each channel is divided by reference,
        so a bad reference channel will spoil the rest of the data.
    """
    pump_paths = rawnav.pump_group_paths(src_file)
    bad_paths = []
    for pump_path in pump_paths:
        perp_path = f'{pump_path}/perp'
        par_path = f'{pump_path}/par'
        ref_path = f'{pump_path}/ref'
        ref_noise = src_file[ref_path].attrs['noise']
        if ref_noise < noise_floor:
            # bad_paths.append(perp_path)
            # bad_paths.append(par_path)
            bad_paths.append(ref_path)
            logger.debug(f'Noise too low: {src_file.filename}::{ref_path}')
            # continue
        perp_noise = src_file[perp_path].attrs['noise']
        par_noise = src_file[par_path].attrs['noise']
        if perp_noise < noise_floor:
            bad_paths.append(perp_path)
            logger.debug(f'Noise too low: {src_file.filename}::{perp_path}')
        if par_noise < noise_floor:
            bad_paths.append(par_path)
            logger.debug(f'Noise too low: {src_file.filename}::{par_path}')
    return bad_paths


def mark_bad_wavelength_groups(src_file: H5File, filtered_paths: List[str]) -> None:
    """Sets the `isbad` attribute of each wavelength group to `True` if the group contains
    one or more path that has been filtered.

    Args:
        src_file (H5File): The file containing the experimental data
        filtered_paths (List[str]): A list of paths to signal datasets that have been caught by
            filters due to noise, etc
    """
    bad_paths = set(filtered_paths)
    for wav_path in rawnav.wavelengths_under_rounds_paths(src_file):
        paths_below = dataset_paths_below(src_file, wav_path)
        if any(p in bad_paths for p in paths_below):
            src_file[wav_path].attrs['isbad'] = True
        else:
            src_file[wav_path].attrs['isbad'] = False
    return


def mark_bad_signals(src_file: H5File, filtered_paths: List[str]) -> None:
    """Sets the `isbad` attribute of each signal to `True` or `False` based on
    previous whether the signal was caught by a filter or not.

    Args:
        src_file (H5File): The file containing the experiment data
        filtered_paths(List[str]): A list containing the paths to signals caught
            by filters
    """
    bad_paths = set(filtered_paths)
    for sig_path in rawnav.all_signal_dataset_paths(src_file):
        if sig_path in bad_paths:
            src_file[sig_path].attrs['isbad'] = True
            logger.debug(f'Marked for filtering: {src_file.filename}::{sig_path}')
        else:
            src_file[sig_path].attrs['isbad'] = False
    return


def report_filter_dry_run(src_file: H5File) -> None:
    """Applies a set of filters to the raw data and generates a report as to how much data
    would be eliminated

    Args:
        src_file (H5File): The file containing the experimental data
    """
    faulty_paths = filter_faulty_groups(src_file)
    faulty_report = generate_filter_report('faulty group', faulty_paths)
    print(faulty_report)
    sigmas = 2.5
    noisy_paths = filter_noise_too_high(src_file, sigmas)
    noise_report = generate_filter_report('Noise too high', noisy_paths, sigmas)
    print(noise_report)
    noise_floor = 1e-6
    noise_too_low = filter_noise_too_low(src_file, noise_floor)
    noise_too_low_report = generate_filter_report('Noise too low', noise_too_low, noise_floor)
    print(noise_too_low_report)
    return


def generate_filter_report(heading: str, paths: List[str], cutoff: float=None) -> str:
    """Generates a formatted report of the results of a filter

    The format of the report is as follows:
        <heading> datasets: <number of paths> (<percentage of all paths>)[ cutoff: <cutoff>]

    Args:
        heading (str): The filter identifier
        paths (List[str]): The paths rejected by the filter
        cutoff (float): If the filter used a cutoff parameter, it can be incorporated
            into the report

    Returns:
        A formatted string containing the results of the filtering operation

    Note:
        Not covered by tests, probably not needed
    """
    total_dsets = 123_648.0
    num_paths = len(paths)
    filtered_percentage = 100 * (num_paths / total_dsets)
    if cutoff is not None:
        report = f'{heading} datasets: {num_paths} ({filtered_percentage:.2f}%) cutoff: {cutoff:.2e}'  # noqa
        return report
    else:
        report = f'{heading} datasets: {num_paths} ({filtered_percentage:.2f}%)'
        return report
