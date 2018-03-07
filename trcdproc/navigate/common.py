from typing import List, Set

from trcdproc.core import H5File, datasets, subgroups


def wavelength_set(organized_file: H5File) -> Set[str]:
    """Walks a reorganized raw-data or dA file and determines the set of all wavelengths present in the file

    Args:
        organized_file (H5File): A file whose data has already been organized into rounds

    Returns:
        The set of wavelengths present in the file
    """
    rounds_root = organized_file['rounds']
    wavelengths = set()
    for rnd_name in subgroups(rounds_root):
        rnd_wavelengths = subgroups(rounds_root[rnd_name])
        wavelengths.update(rnd_wavelengths)
    return wavelengths


def dataset_paths_below(src_file: H5File, group_path: str) -> List[str]:
    """Returns the paths of all datasets below the provided path.

    One use case for this function is returning the paths to each dataset below a group
    that has been deemed to contain bad data

    Args:
        src_file (DataFile): The file containing the experimental data
        group_path (str): The path to the parent group

    Returns:
        The paths of all datasets below the parent group
    """
    dataset_paths = []
    parent_group = src_file[group_path]
    for g_name in subgroups(parent_group):
        dataset_paths += dataset_paths_below(src_file, f'{group_path}/{g_name}')
    for d_name in datasets(parent_group):
        dataset_paths.append(f'{group_path}/{d_name}')
    return dataset_paths
