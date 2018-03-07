from os import remove
from os.path import exists

import h5py
from pytest import fixture

from trcdproc.core import subgroups
from trcdproc.pipelines import restructure_and_merge
from trcdproc.reorganize.common import recursive_copy


@fixture(scope='function')
def filenames_for_merging(clean_raw_data):
    """Generates two identical files to be merged and returns their filenames
    """
    raw_filenames = ['raw1.h5', 'raw2.h5']
    for name in raw_filenames:
        file = h5py.File(name, 'w', libver='latest')
        recursive_copy(clean_raw_data, file)
        file.close()
    yield raw_filenames
    for name in raw_filenames:
        remove(name)


def test_restructure_and_merge_cleans_up_when_finished(filenames_for_merging):
    """Some temporary files are created during the course of restructuring and merging,
    so this test verifies that they are removed after the merge process is completed.
    """
    restructure_and_merge(filenames_for_merging, joined_name='joined.h5')
    remove('joined.h5')
    temp_filenames = ['restructured000.h5', 'restructured001.h5']
    for name in temp_filenames:
        assert not exists(name)


def test_restructure_and_merge_renumbers_correctly(filenames_for_merging):
    """Verifies that the rounds are renumbered correctly during the merge.
    """
    restructure_and_merge(filenames_for_merging, joined_name='joined.h5')
    merged = h5py.File('joined.h5', 'r', libver='latest')
    renamed_rounds = subgroups(merged['rounds'])
    merged.close()
    remove('joined.h5')
    assert len(renamed_rounds) == 6
    for num in range(6):
        expected_round = f'round{num:0>3d}'
        assert expected_round in renamed_rounds


def test_restructure_and_merge_copies_all_data(filenames_for_merging):
    """Verifies that all of the data in the original files makes it into the merged file.
    """
    restructure_and_merge(filenames_for_merging, joined_name='joined.h5')
    merged = h5py.File('joined.h5', 'r', libver='latest')
    for num in range(6):
        rnd = f'round{num:0>3d}'
        for wav in ['76487', '76715']:
            for pump in ['pump', 'nopump']:
                for data in ['time', 'perp', 'par', 'ref']:
                    path = f'rounds/{rnd}/{wav}/{pump}/{data}'
                    assert merged[path].len() == 50_000
    merged.close()
    remove('joined.h5')
