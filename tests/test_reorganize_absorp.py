from pytest import fail

import trcdproc.reorganize.absorp as reorg
from trcdproc.core import subgroups, H5File


def test_copy_rounds_structure_for_delta_a(organized_clean_data: H5File, starts_empty: H5File):
    """Ensures that the groups in the organized data file are correctly copied
    to an empty file that will later store dA data
    """
    reorg.copy_rounds_structure_for_delta_a(organized_clean_data, starts_empty)
    rounds_root = organized_clean_data['rounds']
    for rnd in subgroups(rounds_root):
        for wav in subgroups(rounds_root[rnd]):
            path = f'rounds/{rnd}/{wav}'
            try:
                starts_empty[path]
            except KeyError:
                fail(f'Path not copied into new file: {path}')
