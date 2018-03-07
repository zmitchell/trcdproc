from typing import Generator

from trcdproc.core import H5File, subgroups


def all_signal_dataset_paths(src_file: H5File) -> Generator[str, None, None]:
    """Yields paths to each existing perp/par/ref dataset in a pump/nopump group

    Args:
        src_file (H5File): The file with the experiment data

    Yields:
        Paths to each dataset in the file that isn't a time dataset
    """
    rounds_root = src_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    wavelengths = sorted(subgroups(rounds_root[rounds[0]]))
    for rnd in rounds:
        for wav in wavelengths:
            for pump in ['pump', 'nopump']:
                for channel in ['perp', 'par', 'ref']:
                    path = f'/rounds/{rnd}/{wav}/{pump}/{channel}'
                    try:
                        src_file[path]
                    except KeyError:
                        continue
                    yield path


def pump_group_paths(src_file: H5File) -> Generator[str, None, None]:
    """Yields paths to each existing pump/nopump group under each round

    Args:
        src_file (H5File): The file containing the experimental data

    Note:
        An easier way to generate these paths would be to collect a list of paths
        to all items in a file using the `visit` method, then keep only those paths
        that end with "pump". However, for large files it can take several seconds
        just to iterate through all of those items, so for the sake of speed we
        generate each possible path (not including the faulty ones) and just check
        to make sure it corresponds to a real group before yielding the path.

    Yields:
        Paths to pump/nopump subgroups of each wavelength under each round
    """
    rounds_root = src_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    wavelengths = sorted(subgroups(rounds_root[rounds[0]]))
    for rnd in rounds:
        for wav in wavelengths:
            for pump in ['pump', 'nopump']:
                path = f'/rounds/{rnd}/{wav}/{pump}'
                try:
                    src_file[path]
                except KeyError:
                    continue
                yield path


def wavelengths_under_rounds_paths(src_file: H5File) -> Generator[str, None, None]:
    """Yields paths to each existing wavelength group under each round

    Args:
        src_file (H5File): The file containing the experimental data

    Yields:
        Paths to wavelength subgroups of each round
    """
    rounds_root = src_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    wavelengths = sorted(subgroups(rounds_root[rounds[0]]))
    for rnd in rounds:
        for wav in wavelengths:
            path = f'/rounds/{rnd}/{wav}'
            try:
                src_file[path]
            except KeyError:
                continue
            yield path
