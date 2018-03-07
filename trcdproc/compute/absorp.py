from typing import Callable

import numpy as np
from scipy.interpolate import UnivariateSpline
from logzero import logger

import trcdproc.navigate.raw as rawnav
import trcdproc.navigate.absorp as absnav
import trcdproc.reorganize.absorp as reorg
from trcdproc.navigate.common import wavelength_set
from trcdproc.core import (
    H5File,
    Group,
    Dataset,
    Array,
    subgroups,
    copy_all_attributes,
    DeltaAChannel
)


def should_compute_delta_a(wav_group: Group) -> bool:
    """Returns True if the change in absorption should be computed, or False
    if it shouldn't as in cases where there isn't a pair of pump/nopump datasets

    Args:
        wav_group (Group): The group for a single wavelength in a single round

    Returns:
        True if dA should be computed, False if this computation should be skipped

    Note:
        Not covered by tests yet, probably not needed
    """
    has_isbad_attr = True
    try:
        wav_group.attrs['isbad']
    except KeyError:
        has_isbad_attr = False
    if has_isbad_attr:
        if wav_group.attrs['isbad']:
            return False
    subgroup_names = subgroups(wav_group)
    for name in subgroup_names:
        if 'faulty' in name:
            return False
        else:
            return True


def compute_single_delta_a(old_group: Group, new_group: Group, include_cd: bool=False) -> None:
    """Computes the change in absorption for perpendicular and parallel channels
    and stores the result in the new file.

    Args:
        old_group (Group): The group for a single wavelength in a single round
        new_group (Group): The group in which computed dA values will be stored
        include_cd (bool): Whether to compute the change in absorption of circularly polarized
            light as well

    Note:
        The data is stored as float32, but the computations are carried out as float64 to reduce
        accumulation of rounding errors. This is implemented by creating float64 arrays ahead of
        time and using the `read_direct` method to force the HDF5 library to cast the float32 data
        to float64 on the fly.
    """
    # See the note above for why we create these empty arrays ahead of time
    points = len(old_group['pump/perp'][...])
    perp_with_pump = np.empty(points, dtype=np.float64)
    perp_without_pump = np.empty(points, dtype=np.float64)
    par_with_pump = np.empty(points, dtype=np.float64)
    par_without_pump = np.empty(points, dtype=np.float64)
    ref_with_pump = np.empty(points, dtype=np.float64)
    ref_without_pump = np.empty(points, dtype=np.float64)
    # The read_direct method will automatically cast the data (stored as float32) to float64
    old_group['pump/perp'].read_direct(perp_with_pump)
    old_group['nopump/perp'].read_direct(perp_without_pump)
    old_group['pump/par'].read_direct(par_with_pump)
    old_group['nopump/par'].read_direct(par_without_pump)
    old_group['pump/ref'].read_direct(ref_with_pump)
    old_group['nopump/ref'].read_direct(ref_without_pump)
    # First the perpendicular and parallel channels are divided by their references
    reduced_perp_with_pump = perp_with_pump / ref_with_pump
    reduced_perp_without_pump = perp_without_pump / ref_without_pump
    reduced_par_with_pump = par_with_pump / ref_with_pump
    reduced_par_without_pump = par_without_pump / ref_without_pump
    # Then the ratio of with pump to without pump is computed
    perp_ratio = reduced_perp_with_pump / reduced_perp_without_pump
    par_ratio = reduced_par_with_pump / reduced_par_without_pump
    # Then the logarithm is computed to yield the change in absorption
    with np.errstate(invalid='ignore'):  # ignore runtime warnings about log(negative number)
        delta_a_perp: Array = -1 * np.log10(perp_ratio)
        delta_a_par: Array = -1 * np.log10(par_ratio)
    delta_a_perp = np.nan_to_num(delta_a_perp)  # this is where we handle log(negative number)
    delta_a_par = np.nan_to_num(delta_a_par)
    # Finally, store the result
    new_group.create_dataset('perp', data=delta_a_perp, dtype=np.float32)
    new_group.create_dataset('par', data=delta_a_par, dtype=np.float32)
    new_group['time'] = old_group['pump/time'][...]  # already stored as float32, no need to cast
    # Do the CD calculations
    if include_cd:
        intensity_ratio_with_pump = reduced_perp_with_pump / reduced_par_with_pump
        intensity_ratio_without_pump = reduced_perp_without_pump / reduced_par_without_pump
        delta_a_cd = intensity_ratio_with_pump - intensity_ratio_without_pump
        new_group.create_dataset('cd', data=delta_a_cd, dtype=np.float32)
    return


def compute_single_selective_delta_a(raw_wav_group: Group, new_group: Group, include_cd: bool=False) -> None:  # noqa
    """Computes the change in absorption for perpendicular and parallel channels
    and stores the result in the new file.

    Args:
        raw_wav_group (Group): The group for a single wavelength in a single round
        new_group (Group): The group in which computed dA values will be stored
        include_cd (bool): Whether to compute the change in absorption of circularly polarized
            light as well

    Note:
        The data is stored as float32, but the computations are carried out as float64 to reduce
        accumulation of rounding errors. This is done by creating float64 arrays ahead of
        time and using the `read_direct` method to force the HDF5 library to cast the float32 data
        to float64 on the fly.
    """
    # Bail early if the reference channel is bad, you can't do anything without it
    if raw_wav_group['pump/ref'].attrs['isbad']:
        return
    elif raw_wav_group['nopump/ref'].attrs['isbad']:
        return
    # Assemble a list of signals to compute, removing bad ones
    signal_set = {DeltaAChannel.perp, DeltaAChannel.par}
    if include_cd:
        signal_set.add(DeltaAChannel.cd)
    if raw_wav_group['pump/perp'].attrs['isbad']:
        signal_set.discard(DeltaAChannel.perp)
        signal_set.discard(DeltaAChannel.cd)  # you need perp to calculate CD
    elif raw_wav_group['nopump/perp'].attrs['isbad']:
        signal_set.discard(DeltaAChannel.perp)
        signal_set.discard(DeltaAChannel.cd)
    if raw_wav_group['pump/par'].attrs['isbad']:
        signal_set.discard(DeltaAChannel.par)
        signal_set.discard(DeltaAChannel.cd)  # you need par to calculate CD
    elif raw_wav_group['nopump/par'].attrs['isbad']:
        signal_set.discard(DeltaAChannel.par)
        signal_set.discard(DeltaAChannel.cd)
    signals_to_compute = [item.name for item in signal_set]
    if len(signals_to_compute) == 0:
        return
    # Compute each signal
    points = len(raw_wav_group['pump/ref'][...])
    ref_with_pump = np.empty(points, dtype=np.float64)
    ref_without_pump = np.empty(points, dtype=np.float64)
    raw_wav_group['pump/ref'].read_direct(ref_with_pump)
    raw_wav_group['nopump/ref'].read_direct(ref_without_pump)
    for sig in signals_to_compute:
        if sig == 'cd':
            compute_cd(raw_wav_group, new_group)
            continue
        # Read the float32 data into float64 arrays to avoid accumulation of error
        sig_with_pump = np.empty(points, dtype=np.float64)
        sig_without_pump = np.empty(points, dtype=np.float64)
        raw_wav_group[f'pump/{sig}'].read_direct(sig_with_pump)
        raw_wav_group[f'nopump/{sig}'].read_direct(sig_without_pump)
        # Divide by the reference channel
        reduced_sig_with_pump = sig_with_pump / ref_with_pump
        reduced_sig_without_pump = sig_without_pump / ref_without_pump
        # Take the ratio (with pump) / (without pump)
        sig_ratio = reduced_sig_with_pump / reduced_sig_without_pump
        # Compute the change in absorption, ignoring warnings about log(negative number)
        with np.errstate(invalid='ignore'):
            delta_a = -1 * np.log10(sig_ratio)
        delta_a = np.nan_to_num(delta_a)
        new_group.create_dataset(sig, data=delta_a, dtype=np.float32)  # store as float32
    new_group['time'] = raw_wav_group['pump/time'][...]  # already float32, no need to cast
    return


def compute_cd(raw_wav_group: Group, new_group: Group) -> None:
    """Computes `delta A CD` from the signals contained in the wavelength group.

    Args:
        raw_wav_group (Group): A group containing the `pump`/`nopump` groups for
            a single round
        new_group (Group): The group in which the result will be stored

    Note:
        This function assumes that it will only be called when the perpendicular, parallel,
        and reference channels are known to be good, so it does not check the `isbad`
        attribute of those signals.
    """
    points = len(raw_wav_group['pump/perp'][...])
    perp_with_pump = np.empty(points, dtype=np.float64)
    perp_without_pump = np.empty(points, dtype=np.float64)
    par_with_pump = np.empty(points, dtype=np.float64)
    par_without_pump = np.empty(points, dtype=np.float64)
    ref_with_pump = np.empty(points, dtype=np.float64)
    ref_without_pump = np.empty(points, dtype=np.float64)
    # The read_direct method will automatically cast the data (stored as float32) to float64
    raw_wav_group['pump/perp'].read_direct(perp_with_pump)
    raw_wav_group['nopump/perp'].read_direct(perp_without_pump)
    raw_wav_group['pump/par'].read_direct(par_with_pump)
    raw_wav_group['nopump/par'].read_direct(par_without_pump)
    raw_wav_group['pump/ref'].read_direct(ref_with_pump)
    raw_wav_group['nopump/ref'].read_direct(ref_without_pump)
    # First the perpendicular and parallel channels are divided by their references
    reduced_perp_with_pump = perp_with_pump / ref_with_pump
    reduced_perp_without_pump = perp_without_pump / ref_without_pump
    reduced_par_with_pump = par_with_pump / ref_with_pump
    reduced_par_without_pump = par_without_pump / ref_without_pump
    # Compute the ratios with and without pump
    ratio_with_pump = reduced_perp_with_pump / reduced_par_with_pump
    ratio_without_pump = reduced_perp_without_pump / reduced_par_without_pump
    # Compute and save the CD data
    cd = ratio_with_pump - ratio_without_pump
    new_group.create_dataset('cd', data=cd, dtype=np.float32)
    return


def compute_all_delta_a(old_file: H5File, new_file: H5File, include_cd: bool=False) -> None:
    """Computes the change in absorption for all cases where valid data exists
    and stores the results in a new file

    Args:
        old_file (H5File): The file in which the raw data is stored
        new_file (H5File): The file in which the computed values will be stored
        include_cd (bool): Whether to compute the change in absorption of circularly polarized
            light as well
    """
    reorg.copy_rounds_structure_for_delta_a(old_file, new_file)
    old_rounds_root = old_file['rounds']
    new_rounds_root = new_file['rounds']
    for rnd in subgroups(old_rounds_root):
        for wav in subgroups(old_rounds_root[rnd]):
            wav_group_path = f'{rnd}/{wav}'
            old_group = old_rounds_root[wav_group_path]
            if not should_compute_delta_a(old_group):
                continue
            new_group = new_rounds_root[wav_group_path]
            compute_single_delta_a(old_group, new_group, include_cd)
    return


def compute_only_good_delta_a(old_file: H5File, new_file: H5File, include_cd: bool=False) -> None:
    """Computes the change in absorption only for wavelength groups that are not marked as 'bad'

    Args:
        old_file (H5File): The file containing the raw data
        new_file (H5File): The file in which the change in absorption will be stored
        include_cd (bool): Whether to compute the change in absorption of circularly polarized
            light as well
    """
    for wav_path in rawnav.wavelengths_under_rounds_paths(old_file):
        if old_file[wav_path].attrs['isbad']:
            continue
        new_wav_group = new_file.require_group(wav_path)
        compute_single_delta_a(old_file[wav_path], new_wav_group, include_cd)
    return


def compute_delta_a_selectively(old_file: H5File, new_file: H5File, include_cd: bool=False) -> None:
    for wav_path in rawnav.wavelengths_under_rounds_paths(old_file):
        if 'faulty1' in subgroups(old_file[wav_path]):
            continue
        new_wav_group = new_file.require_group(wav_path)
        # logger.debug(f'Computing dA at path: {wav_path}')
        compute_single_selective_delta_a(old_file[wav_path], new_wav_group, include_cd)
    return


def adjust_baseline(src_file: H5File, dest_file: H5File, include_cd: bool=False) -> None:
    """Adds a constant value to every point such that the average of the points before
    the laser pulse arrives is zero

    Args:
        src_file (H5File): The file containing the unadjusted data
        dest_file (H5File): The file containing the adjusted data
        include_cd (bool): Whether to adjust the baseline of the CD data as well
    """
    channels = [DeltaAChannel.perp, DeltaAChannel.par, DeltaAChannel.cd]
    for channel in channels:
        for signal_path in absnav.paths_for_signal(src_file, channel):
            signal_data = np.empty(50_000, dtype=np.float64)
            src_file[signal_path].read_direct(signal_data)
            mean = signal_data[0:975].mean()
            signal_data -= mean
            group_path = src_file[signal_path].parent.name
            group = dest_file.require_group(group_path)
            group.create_dataset(channel.name, data=signal_data, dtype=np.float32)
            try:
                group['time']
            except KeyError:
                group['time'] = src_file[group_path + '/time'][...]
    # for wav_path in rawnav.wavelengths_under_rounds_paths(src_file):
    #     perp_path = f'{wav_path}/perp'
    #     par_path = f'{wav_path}/par'
    #     perp_data = np.empty(50_000, dtype=np.float64)
    #     par_data = np.empty(50_000, dtype=np.float64)
    #     src_file[perp_path].read_direct(perp_data)
    #     src_file[par_path].read_direct(par_data)
    #     perp_mean = perp_data[0:975].mean()
    #     par_mean = par_data[0:975].mean()
    #     perp_data -= perp_mean
    #     par_data -= par_mean
    #     dest_file.require_group(wav_path)
    #     dest_file[wav_path].create_dataset('perp', data=perp_data, dtype=np.float32)
    #     dest_file[wav_path].create_dataset('par', data=par_data, dtype=np.float32)
    #     if include_cd:
    #         cd_path = f'{wav_path}/cd'
    #         cd_data = np.empty(50_000, dtype=np.float64)
    #         src_file[cd_path].read_direct(cd_data)
    #         cd_mean = cd_data[0:975].mean()
    #         cd_data -= cd_mean
    #         dest_file[wav_path].create_dataset('cd', data=cd_data, dtype=np.float32)
    #     dest_file[wav_path + '/time'] = src_file[wav_path + '/time'][...]
    return


def average_delta_a(src_file: H5File, include_cd: bool=False) -> None:
    """Averages the change in absorption at each wavelength and stores the results in a new
    top-level group called 'averaged'

    Args:
        src_file (H5File): The file containing the data to be averaged
        include_cd (bool): Whether to average the CD data as well

    Note:
        This function traverses the data using the top level wavelength group, which is no longer
        added to the data by default.

    Note:
        Not covered by tests yet

    Warning:
        Deprecated, use `average_rounds` instead
    """
    src_file.create_group('averaged')
    wav_root = src_file['wavelengths']
    time_data = np.empty(50_000, dtype=np.float32)
    wav_root[f'76487/round000/time'].read_direct(time_data)
    for wav in subgroups(wav_root):
        perp_summed = np.empty(50_000, dtype=np.float64)
        par_summed = np.empty(50_000, dtype=np.float64)
        count = 0
        for rnd in subgroups(wav_root[wav]):
            perp_path = f'{wav}/{rnd}/perp'
            par_path = f'{wav}/{rnd}/par'
            perp_data = np.empty(50_000, dtype=np.float64)
            par_data = np.empty(50_000, dtype=np.float64)
            wav_root[perp_path].read_direct(perp_data)
            wav_root[par_path].read_direct(par_data)
            perp_summed += perp_data
            par_summed += par_data
            count += 1
        perp_avg = perp_summed / count
        par_avg = par_summed / count
        wav_avg_group = src_file.create_group(f'averaged/{wav}')
        wav_avg_group.create_dataset('perp', data=perp_avg, dtype=np.float32)
        wav_avg_group.create_dataset('par', data=par_avg, dtype=np.float32)
        wav_avg_group.create_dataset('time', data=time_data)
    return


def average_rounds(src_file: H5File, include_cd: bool=False) -> None:
    """Averages the change in absorption at each wavelength and stores the result in a new
    top-level group named `averaged`

    Args:
        src_file (H5File): The file containing the experiment data
        include_cd (bool): Whether to average the CD data as well
    """

    all_wavelengths = wavelength_set(src_file)
    sums = {  # this will hold the running total for each wavelength
        wav: {
            'perp': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            },
            'par': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            },
            'cd': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            }
        } for wav in all_wavelengths
    }
    signals = ['perp', 'par']
    if include_cd:
        signals.append('cd')
    # store the totals and counts for each wavelength and signal channel
    time_data = np.empty(50_000, dtype=np.float32)
    time_data_loaded = False
    rounds_root = src_file['rounds']
    for rnd in subgroups(rounds_root):
        for wav in subgroups(rounds_root[rnd]):
            wav_path = f'{rnd}/{wav}'
            if not time_data_loaded:  # only load time data once per wavelength group
                rounds_root[wav_path + '/time'].read_direct(time_data)
                time_data_loaded = True
            for sig in signals:
                sig_path = f'{rnd}/{wav}/{sig}'
                sig_data = np.empty(50_000, dtype=np.float64)
                rounds_root[sig_path].read_direct(sig_data)
                sums[wav][sig]['sum'] += sig_data
                sums[wav][sig]['count'] += 1
    # store the averages
    avg_root = src_file.create_group('averaged')
    for wav in sums.keys():
        wav_group = avg_root.create_group(wav)
        for sig in signals:
            averaged = sums[wav][sig]['sum'] / sums[wav][sig]['count']
            wav_group.create_dataset(sig, data=averaged, dtype=np.float32)
            wav_group[sig].attrs['count'] = sums[wav][sig]['count']
    return


def average_selectively(src_file: H5File, include_cd: bool=False) -> None:
    """Averages the change in absorption at each wavelength and stores the result in a new
    top-level group named `averaged`

    Args:
        src_file (H5File): The file containing the experiment data
        include_cd (bool): Whether to average the CD data as well

    Note:
        This function assumes that individual signals may be missing due to being filtered at
        an earlier stage.
    """

    all_wavelengths = wavelength_set(src_file)
    sums = {  # this will hold the running total for each wavelength
        wav: {
            'perp': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            },
            'par': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            },
            'cd': {
                'sum': np.zeros(50_000, dtype=np.float64),
                'count': 0,
            }
        } for wav in all_wavelengths
    }
    signals = ['perp', 'par']
    if include_cd:
        signals.append('cd')
    # store the totals and counts for each wavelength and signal channel
    time_data = np.empty(50_000, dtype=np.float32)
    time_data_loaded = False
    rounds_root = src_file['rounds']
    for rnd in subgroups(rounds_root):
        for wav in subgroups(rounds_root[rnd]):
            wav_path = f'{rnd}/{wav}'
            logger.info(f'Collecting signals from {wav_path}')
            if not time_data_loaded:  # only load time data once per wavelength group
                rounds_root[wav_path + '/time'].read_direct(time_data)
                time_data_loaded = True
            for sig in signals:
                sig_path = f'{rnd}/{wav}/{sig}'
                try:
                    rounds_root[sig_path]
                except KeyError:
                    continue
                sig_data = np.empty(50_000, dtype=np.float64)
                rounds_root[sig_path].read_direct(sig_data)
                sums[wav][sig]['sum'] += sig_data
                sums[wav][sig]['count'] += 1
    # store the averages
    avg_root = src_file.create_group('averaged')
    for wav in sums.keys():
        wav_group = avg_root.create_group(wav)
        for sig in signals:
            signal_path = f'{wav}/{sig}'
            logger.info(f'Creating averaged dataset {signal_path}')
            averaged = sums[wav][sig]['sum'] / sums[wav][sig]['count']
            wav_group.create_dataset(sig, data=averaged, dtype=np.float32)
            wav_group[sig].attrs['count'] = sums[wav][sig]['count']
        wav_group.create_dataset('time', data=time_data, dtype=np.float32)
    return


def collapse_all(old_file: H5File, new_file: H5File, n: int, include_cd: bool=False) -> None:
    """Collapses all of the datasets in the source file and stores the result
    in the destination file

    This operation takes `N` adjacent points and replaces those `N` points with a single point
    whose value is the their average. This increases the SNR by a factor of `sqrt(N)`, reduces
    file size by a factor of `N`, and increases the time between points by a factor of `N`.

    Args:
        old_file (H5File): The file containing the original data
        new_file (H5File): The new, empty file that will be filled with the collapsed data
        n (int): The number of points to collapse together
        include_cd (bool): Whether to include the CD data in the new file

    Note:
        Not covered by tests yet
    """
    reorg.copy_rounds_structure_for_delta_a(old_file, new_file)
    old_rounds_root = old_file['rounds']
    new_rounds_root = new_file['rounds']
    logger.info(f'Collapsing {old_file.filename} into {new_file.filename} with a window of {n} points')  # noqa
    for rnd_name in subgroups(old_rounds_root):
        for wav_name in subgroups(old_rounds_root[rnd_name]):
            wav_path = f'{rnd_name}/{wav_name}'
            wav_group = old_rounds_root[wav_path]
            logger.debug(f'Collapsing the data in group {wav_group.name}')
            time = collapse_single(wav_group['time'], n)
            perp = collapse_single(wav_group['perp'], n)
            par = collapse_single(wav_group['par'], n)
            new_rounds_root[wav_path].create_dataset('time', data=time, dtype=np.float32)
            new_rounds_root[wav_path].create_dataset('perp', data=perp, dtype=np.float32)
            new_rounds_root[wav_path].create_dataset('par', data=par, dtype=np.float32)
            if include_cd:
                cd = collapse_single(wav_group['cd'], n)
                new_rounds_root[wav_path].create_dataset('cd', data=cd, dtype=np.float32)
    copy_all_attributes(old_file, new_file)
    return


def collapse_single(src_data: Dataset, n: int) -> Array:
    """Replaces a set of `N` adjacent points with a single point whose value is the
    average of the `N` points.

    Args:
        src_data (Dataset): The data to collapse
        n (int): The number of points to collapse

    Returns:
        The collapsed data
    """
    points = len(src_data)
    array = np.empty(points, dtype=np.float64)
    src_data.read_direct(array)  # cast float32 to float64 for computations
    max_steps = points // n
    collapsed = np.empty(max_steps, dtype=np.float64)
    dest_location = 0
    src_location = 0
    while dest_location < max_steps:
        collapsed[dest_location] = array[src_location:(src_location + n)].mean()
        src_location += n
        dest_location += 1
    return collapsed


def store_splined(src: Group, dest: Group, include_cd: bool=False) -> None:
    """Stores the splined version of the data from the source group in the destination group

    Args:
        src (Group): The group containing the original data
        dest (Group): The group in which to store the splined data
        include_cd (bool): Whether to store the splined CD data

    Note:
        Not covered by tests yet
    """
    points = len(src['time'][...])
    time_data = np.empty(points, dtype=np.float64)
    perp_data = np.empty(points, dtype=np.float64)
    par_data = np.empty(points, dtype=np.float64)
    src['time'].read_direct(time_data)  # read float32 data into float64 array for precision
    src['perp'].read_direct(perp_data)
    src['par'].read_direct(par_data)
    perp_splined = noise_spline_data(time_data, perp_data)
    par_splined = noise_spline_data(time_data, par_data)
    dest.create_dataset('time', data=src['time'][...])  # already stored as float32
    dest.create_dataset('perp', data=perp_splined, dtype=np.float32)  # store as float32 for space
    dest.create_dataset('par', data=par_splined, dtype=np.float32)
    if include_cd:
        cd_data = np.empty(points, dtype=np.float64)
        src['cd'].read_direct(cd_data)
        cd_splined = noise_spline_data(time_data, cd_data)
        dest.create_dataset('cd', data=cd_splined, dtype=np.float32)
    return


def noise_spline_data(time: Array, absorption: Array) -> Array:
    """Returns the absorption data as calculated by the smoothed spline

    Args:
        time (Array): The time values for the absorption data
        absorption (Array): The absorption data to fit with a spline

    Returns:
        spline_data (Array): The absorption data constructed from the spline

    Note:
        Not covered by tests yet
    """
    spline = UnivariateSpline(time, absorption)
    spline_data = spline(time)
    return spline_data


def delta_a_heatmap(file: H5File, func: Callable[[Array, Array], np.float64],
                    channel: DeltaAChannel) -> Array:
    """Generates a (number of rounds)x(number of wavelengths) array by visiting each wavelength
    group in the file. The value at each position in the array is computed by
    the function provided.

    Args:
        file (H5File): The file to generate the heatmap data from
        func (Callable): A function that will be used to compute the heatmap values
        channel (DeltaAChannel): The dA channel to use

    Returns:
        A (number of rounds)x(number of wavelengths) array that contains the heatmap pixels

    Note:
        The "func" argument must have the following signature:
            func(time: Array, signal: Array) -> np.float64
    """
    rounds_root = file['rounds']
    rounds = subgroups(rounds_root)
    unsorted_wavelength_set = wavelength_set(file)
    wavelengths = sorted(list(unsorted_wavelength_set))
    pixels = np.ndarray(shape=(len(rounds), len(wavelengths)), dtype=np.float64)
    for i in range(len(rounds)):
        for j in range(len(wavelengths)):
            group_path = f'{rounds[i]}/{wavelengths[j]}'
            try:
                group = rounds_root[group_path]
            except KeyError:
                pixels[i, j] = 0
                continue
            time_dset = group['time']
            signal_dset = group[f'{channel.value}']
            points = len(group['time'][...])
            # cast float32 data to float64 for computations to avoid rounding errors
            time_data = np.empty(points, dtype=np.float64)
            signal_data = np.empty(points, dtype=np.float64)
            time_dset.read_direct(time_data)
            signal_dset.read_direct(signal_data)
            pixels[i, j] = func(time_data, signal_data)
    return pixels
