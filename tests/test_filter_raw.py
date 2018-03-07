from os import remove

import h5py
import numpy as np
from pytest import fixture

from trcdproc.filter.raw import (
    filter_faulty_groups,
    filter_noise_too_low,
    filter_noise_too_high,
    mark_bad_wavelength_groups,
)


@fixture(scope='function')
def file_for_filtering_faulty_data():
    filename = 'filter_faulty.h5'
    file = h5py.File(filename, 'w', libver='latest')
    dummy_data = np.zeros(50_000, dtype=np.float64)  # the actual data doesn't matter for this test
    faulty_group = file.require_group('rounds/round000/76487/faulty1')  # should be filtered
    faulty_group.create_dataset('time', data=dummy_data)
    faulty_group.create_dataset('perp', data=dummy_data)
    faulty_group.create_dataset('par', data=dummy_data)
    faulty_group.create_dataset('ref', data=dummy_data)
    valid_group = file.require_group('rounds/round000/76715/pump')  # should be ignored by filter
    valid_group.create_dataset('time', data=dummy_data)
    valid_group.create_dataset('perp', data=dummy_data)
    valid_group.create_dataset('par', data=dummy_data)
    valid_group.create_dataset('ref', data=dummy_data)
    yield file
    file.close()
    remove(filename)


@fixture(scope='function')
def file_for_general_filtering():
    """Constructs a file in which some data will be picked up by `filter_noise_too_low`, and other
    data will be picked up by `filter_noise_too_high`.

    It is expected that `perp` will be picked up by the low noise filter, while `par` and `ref` will
    be picked up by the high noise filter. The low noise filter only uses the value of the
    individual noise whereas the high noise filter uses the channel's mean noise, the
    standard deviation of the channel's noise, and the value of the noise in the individual dataset.

    Another thing to keep in mind is that if the reference channel is bad, all three channels are
    thrown out, so the reference channel shouldn't be pick up by either filter.
    """
    filename = 'filter_noise.h5'
    file = h5py.File(filename, 'w', libver='latest')
    dummy_data = np.zeros(50_000, dtype=np.float64)  # the actual data isn't used for filtering
    group = file.require_group('rounds/round000/76487/pump')
    # store the actual data
    group.create_dataset('time', data=dummy_data)
    group.create_dataset('perp', data=dummy_data)
    group.create_dataset('par', data=dummy_data)
    group.create_dataset('ref', data=dummy_data)
    # set the individual noise attributes
    file['/rounds/round000/76487/pump/perp'].attrs['noise'] = 0.0  # caught by low noise filter
    file['/rounds/round000/76487/pump/par'].attrs['noise'] = 5.0  # caught by high noise filter
    file['/rounds/round000/76487/pump/ref'].attrs['noise'] = 2.0  # not filtered
    # set the file-level noise attributes
    file.attrs['perp_noise_mean'] = 0.0
    file.attrs['perp_noise_std_dev'] = 5.0  # doesn't matter, mean noise = individual noise for perp
    file.attrs['par_noise_mean'] = 1.0
    file.attrs['par_noise_std_dev'] = 0.1  # make it small so the high noise filter is strict
    file.attrs['ref_noise_mean'] = 1.0
    file.attrs['ref_noise_std_dev'] = 5.0  # make this large so the high noise filter is lax
    yield file
    file.close()
    remove(filename)


def test_filter_faulty_groups_filters_faulty_and_ignores_valid(file_for_filtering_faulty_data):
    """Verifies that the `faulty` filter picks up groups that contain faulty data,
    but ignores the groups that are valid.
    """
    faulty_paths = filter_faulty_groups(file_for_filtering_faulty_data)
    assert len(faulty_paths) == 3
    path_root = '/rounds/round000/76487/faulty1/'
    for sig in ['perp', 'par', 'ref']:
        assert f'{path_root}{sig}' in faulty_paths


def test_filter_data_with_noise_too_low(file_for_general_filtering):
    """Verifies that datasets with unreasonably low noise are filtered out
    """
    max_noise_allowed = 1.0
    filtered_paths = filter_noise_too_low(file_for_general_filtering, max_noise_allowed)
    assert filtered_paths == ['/rounds/round000/76487/pump/perp']


def test_filter_data_with_noise_too_high(file_for_general_filtering):
    """Verifies that datasets with noise larger than some specified value are filtered out
    """
    filter_size = 1.0  # noise more than 1.0 std. dev. from the mean will be filtered
    filtered_paths = filter_noise_too_high(file_for_general_filtering, filter_size)
    assert filtered_paths == ['/rounds/round000/76487/pump/par']


def test_bad_wavelength_groups_are_marked_bad(file_for_general_filtering):
    """Verifies that a wavelength group is marked as "bad" if it contains a dataset
    that has been filtered.
    """
    filtered_paths = ['/rounds/round000/76487/pump/par']
    fake_paths = ['foo']
    wav_path = '/rounds/round000/76487'
    mark_bad_wavelength_groups(file_for_general_filtering, fake_paths)
    assert not file_for_general_filtering[wav_path].attrs['isbad']
    mark_bad_wavelength_groups(file_for_general_filtering, filtered_paths)
    assert file_for_general_filtering[wav_path].attrs['isbad']
