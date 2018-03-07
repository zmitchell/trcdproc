from enum import Enum
from typing import Any, List

import h5py
import numpy as np


H5File = h5py.File
Group = h5py.Group
Dataset = h5py.Dataset
Array = np.ndarray


class InputChannel(Enum):
    time = 'time'
    perp = 'perp'
    par = 'par'
    ref = 'ref'


class DeltaAChannel(Enum):
    time = 'time'
    perp = 'perp'
    par = 'par'
    cd = 'cd'


class PumpStatus(Enum):
    present = 'pump'
    absent = 'nopump'


def subgroups(group: Group, fullpath=False) -> List[str]:
    """Returns the groups that are directly below the specified group as a list of strings

    Args:
        group (Group): The group whose subgroups you want to query
        fullpath (bool): A boolean parameter indicating that the full paths of the
            groups should be returned i.e. '/foo/bar/baz' vs. 'baz'

    Note:
        This may become a generator rather than a function in the future

    Returns:
        The names of any groups that are directly below the specified group
    """
    subgroups = [key for key in group.keys() if isinstance(group[key], Group)]
    if fullpath:
        full_paths = [f'{group.name}/{s}' for s in subgroups]
        subgroups = full_paths
    return subgroups


def datasets(group: Group, fullpath=False) -> List[str]:
    """Returns the datasets that are directly below the specified group as a list of strings

    Args:
        group (Group): The group whose datasets you want to query
        fullpath (bool): A boolean parameter indicating that the full paths of the
            datasets should be returned i.e. '/spectrum6/FMO_dataset' vs. 'FMO_dataset'

    Note:
        This may become a generator in the future

    Returns:
        The names of the datasets contained in the group
    """
    datasets = [k for k in group.keys() if isinstance(group[k], Dataset)]
    if fullpath:
        full_paths = [f'{group.name}/{d}' for d in datasets]
        datasets = full_paths
    return datasets


def add_name_attributes(file: H5File) -> None:
    """Adds the filename as an attribute to the root group, then adds the group or dataset
    name as an attribute to the corresponding objects recursively

    Args:
        file (DataFile): The file to add attributes to

    Note:
        Not covered by tests yet
    """
    filename = file.filename[0:-3]
    file.attrs['name'] = filename
    file.attrs['filename'] = filename
    recursive_add_name_attribute(file)
    return


def recursive_add_name_attribute(group: Group) -> None:
    """Adds the name of the group as an attribute called 'name' to each group or dataset

    The Group.name and Dataset.name properties return the full path to the object, rather than
    just the name of the object. Setting the name as an attribute of the object makes it trivial
    to programmatically generate names of plots, log messages about certain objects, etc.

    Args:
        group (Group): The group to recursively add name attributes to

    Note:
        Not covered by tests yet
    """
    for dataset_name in datasets(group):
        group[dataset_name].attrs['name'] = dataset_name
    for group_name in subgroups(group):
        group[group_name].attrs['name'] = group_name
        recursive_add_name_attribute(group[group_name])
    return


def recursive_set_attribute(group: Group, attr: str, value: Any) -> None:
    """Sets the specified attribute to the provided value for all descendants
    of the group (groups and datasets alike)

    Args:
        group (Group): The group whose descendants will have their attributes set
        attr (str): The attribute to set
        value (Any): The value to set the attribute to

    Note:
        Not covered by tests yet
    """
    for d in datasets(group):
        group[d].attrs[attr] = value
    for sg in subgroups(group):
        group[sg].attrs[attr] = value
        recursive_set_attribute(group[sg], attr, value)
    return


def copy_all_attributes(old_file: H5File, new_file: H5File) -> None:
    """Recursively copies all of the attributes from the old file to the new file

    Args:
        old_file (DataFile): the source file
        new_file (DataFile): the destination file

    Note:
        The two files must have the same group and dataset structure

    Note:
        Not covered by tests yet
    """
    def copy_func(_: str, src: Group) -> None:
        src_path = src.name
        for item in src.attrs.keys():
            if item == 'filename':
                continue
            new_file[src_path].attrs[item] = src.attrs[item]
        return
    old_file.visititems(copy_func)
    recursive_set_attribute(new_file, 'filename', new_file.filename[0:-3])
    return


def count_signal_datasets(group: Group) -> int:
    """Recursively counts the number of signal, i.e. not time, datasets in the group

    Args:
        group (Group): A group potentially containing a mix of datasets and subgroups

    Returns:
        The total number of signal datasets contained by the group at any depth
    """
    count = 0
    for d_name in datasets(group):
        if 'time' in d_name:
            continue
        count += 1
    for g_name in subgroups(group):
        count += count_signal_datasets(group[g_name])
    return count
