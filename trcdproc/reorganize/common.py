import numpy as np

from trcdproc.core import datasets, subgroups, Group


def recursive_copy(old_parent: Group, new_parent: Group) -> None:
    """Copies the contents of the old parent group to the new parent group

    Args:
        old_parent (Group): The group whose contents will be copied
        new_parent (Group): The group that will be copied into
    """
    if len(subgroups(old_parent)) == 0:
        for dset_name in datasets(old_parent):
            new_parent.create_dataset(dset_name, data=old_parent[dset_name][...], dtype=np.float32)
        return
    for group_name in subgroups(old_parent):
        new_parent.create_group(group_name)
        recursive_copy(old_parent[group_name], new_parent[group_name])
    return
