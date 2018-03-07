from os import remove
from typing import Dict

import h5py
import numpy as np

import trcdproc.reorganize.raw as reorg
from trcdproc.core import subgroups, H5File, Group


def test_correct_rounds_created_in_starts_empty(clean_raw_data, starts_empty):
    """Ensures that `spectrum1` gets created in the empty file as `round001`.

    Args:
        clean_raw_data (H5File): An HDF5 file containing clean test data
        starts_empty (H5File): An empty file for testing the creation of rounds
    """
    reorg.create_rounds_from_spectra(clean_raw_data, starts_empty)
    rounds = subgroups(starts_empty['rounds'])
    for rnd in ['round001', 'round002', 'round003']:
        assert rnd in rounds


def test_spectrum_name_map():
    """Ensures that the map of `spectrum00X` names is generated correctly from
    a list of `spectrumX` names
    """
    original_spec_names = ['spectrum1', 'spectrum10', 'spectrum100']
    mapped_spec_names = ['spectrum001', 'spectrum010', 'spectrum100']
    spec_map: Dict[str, str] = reorg.make_spectrum_map(original_spec_names)
    for spec_name in spec_map.keys():
        assert spec_name in mapped_spec_names


def test_create_wavelength_groups_under_one_round(clean_raw_data, starts_empty):
    """Ensures that all of the wavelengths present in a given spectrum of the original file
    end up in the new file
    """
    old_group: Group = clean_raw_data['spectrum1']
    reorg.create_wavelength_groups_for_one_round(old_group, starts_empty)
    # Note that `starts_empty` is no longer empty at this point
    wavelength_groups_created = subgroups(starts_empty)
    assert len(wavelength_groups_created) == 2
    for wav in ['76487', '76715']:
        assert wav in wavelength_groups_created


def test_create_wavelength_groups_under_all_rounds(clean_raw_data, has_rounds_groups):
    """Ensures that all of the wavelength groups are created correctly in the new file
    """
    reorg.create_wavelength_groups_for_all_rounds(clean_raw_data, has_rounds_groups)
    rounds = ['round001', 'round002', 'round003']
    for rnd_name in rounds:
        wavelengths_created = subgroups(has_rounds_groups[f'rounds/{rnd_name}'])
        assert len(wavelengths_created) == 2
        assert '76487' in wavelengths_created
        assert '76715' in wavelengths_created


def test_dataset_name_pairing_with_clean_data():
    """Ensures that the dataset names are paired correctly under the best possible conditions
    """
    dataset_names = ['FMO_176487', 'FMO_276487', 'FMO_176715', 'FMO_276715']
    pairs = reorg.pair_dataset_names(dataset_names)
    # Not sure which order the two dataset names arrive in, so allow for both possibilities
    expected_pairs = [
        ('76487', 'FMO_176487', 'FMO_276487'),
        ('76487', 'FMO_276487', 'FMO_176487'),
        ('76715', 'FMO_176715', 'FMO_276715'),
        ('76715', 'FMO_276715', 'FMO_176715'),
    ]
    assert len(pairs) == 2
    for pair in pairs:
        assert pair in expected_pairs


def test_dataset_name_pairing_with_faulty_data():
    """Ensures that the dataset names are paired correctly when
    one unpaired dataset is present
    """
    dataset_names = ['FMO_176487', 'FMO_276487', 'FMO_176715', 'FMO_276715', 'FMO_176940']
    pairs = reorg.pair_dataset_names(dataset_names)
    expected_pairs = [
        ('76487', 'FMO_176487', 'FMO_276487'),
        ('76487', 'FMO_276487', 'FMO_176487'),
        ('76715', 'FMO_176715', 'FMO_276715'),
        ('76715', 'FMO_276715', 'FMO_176715'),
        ('76940', 'FMO_176940')
    ]
    assert len(pairs) == 3
    for pair in pairs:
        assert pair in expected_pairs


def test_copying_datasets(faulty_raw_data: H5File):
    """Ensures that all of the data in the original file makes it into the new file
    """
    destination = h5py.File('destination.h5', 'w', libver='latest')
    reorg.create_rounds_from_spectra(faulty_raw_data, destination)
    reorg.create_wavelength_groups_for_all_rounds(faulty_raw_data, destination)
    # We need to create a file with the expected structure to compare against the copied file
    expected = h5py.File('expected.h5', 'w', libver='latest')
    expected_rounds_root = expected.create_group('rounds')
    opts = {'shape': (50_000,), 'dtype': np.float64}
    # round001 @ 76487
    expected_rounds_root.require_dataset('round001/76487/pump/time', **opts)
    expected_rounds_root.require_dataset('round001/76487/pump/perp', **opts)
    expected_rounds_root.require_dataset('round001/76487/pump/par', **opts)
    expected_rounds_root.require_dataset('round001/76487/pump/ref', **opts)
    expected_rounds_root.require_dataset('round001/76487/nopump/time', **opts)
    expected_rounds_root.require_dataset('round001/76487/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round001/76487/nopump/par', **opts)
    expected_rounds_root.require_dataset('round001/76487/nopump/ref', **opts)
    # round001 @ 76715
    expected_rounds_root.require_dataset('round001/76715/pump/time', **opts)
    expected_rounds_root.require_dataset('round001/76715/pump/perp', **opts)
    expected_rounds_root.require_dataset('round001/76715/pump/par', **opts)
    expected_rounds_root.require_dataset('round001/76715/pump/ref', **opts)
    expected_rounds_root.require_dataset('round001/76715/nopump/time', **opts)
    expected_rounds_root.require_dataset('round001/76715/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round001/76715/nopump/par', **opts)
    expected_rounds_root.require_dataset('round001/76715/nopump/ref', **opts)
    # round001 @ 76940
    expected_rounds_root.require_dataset('round001/76940/faulty1/time', **opts)
    expected_rounds_root.require_dataset('round001/76940/faulty1/perp', **opts)
    expected_rounds_root.require_dataset('round001/76940/faulty1/par', **opts)
    expected_rounds_root.require_dataset('round001/76940/faulty1/ref', **opts)
    # round002 @ 76487
    expected_rounds_root.require_dataset('round002/76487/pump/time', **opts)
    expected_rounds_root.require_dataset('round002/76487/pump/perp', **opts)
    expected_rounds_root.require_dataset('round002/76487/pump/par', **opts)
    expected_rounds_root.require_dataset('round002/76487/pump/ref', **opts)
    expected_rounds_root.require_dataset('round002/76487/nopump/time', **opts)
    expected_rounds_root.require_dataset('round002/76487/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round002/76487/nopump/par', **opts)
    expected_rounds_root.require_dataset('round002/76487/nopump/ref', **opts)
    # round002 @ 76715
    expected_rounds_root.require_dataset('round002/76715/pump/time', **opts)
    expected_rounds_root.require_dataset('round002/76715/pump/perp', **opts)
    expected_rounds_root.require_dataset('round002/76715/pump/par', **opts)
    expected_rounds_root.require_dataset('round002/76715/pump/ref', **opts)
    expected_rounds_root.require_dataset('round002/76715/nopump/time', **opts)
    expected_rounds_root.require_dataset('round002/76715/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round002/76715/nopump/par', **opts)
    expected_rounds_root.require_dataset('round002/76715/nopump/ref', **opts)
    # round003 - identical to round002
    expected_rounds_root.require_dataset('round003/76487/pump/time', **opts)
    expected_rounds_root.require_dataset('round003/76487/pump/perp', **opts)
    expected_rounds_root.require_dataset('round003/76487/pump/par', **opts)
    expected_rounds_root.require_dataset('round003/76487/pump/ref', **opts)
    expected_rounds_root.require_dataset('round003/76487/nopump/time', **opts)
    expected_rounds_root.require_dataset('round003/76487/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round003/76487/nopump/par', **opts)
    expected_rounds_root.require_dataset('round003/76487/nopump/ref', **opts)
    expected_rounds_root.require_dataset('round003/76715/pump/time', **opts)
    expected_rounds_root.require_dataset('round003/76715/pump/perp', **opts)
    expected_rounds_root.require_dataset('round003/76715/pump/par', **opts)
    expected_rounds_root.require_dataset('round003/76715/pump/ref', **opts)
    expected_rounds_root.require_dataset('round003/76715/nopump/time', **opts)
    expected_rounds_root.require_dataset('round003/76715/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round003/76715/nopump/par', **opts)
    expected_rounds_root.require_dataset('round003/76715/nopump/ref', **opts)
    # round010 @ 76487
    expected_rounds_root.require_dataset('round010/76487/pump/time', **opts)
    expected_rounds_root.require_dataset('round010/76487/pump/perp', **opts)
    expected_rounds_root.require_dataset('round010/76487/pump/par', **opts)
    expected_rounds_root.require_dataset('round010/76487/pump/ref', **opts)
    expected_rounds_root.require_dataset('round010/76487/nopump/time', **opts)
    expected_rounds_root.require_dataset('round010/76487/nopump/perp', **opts)
    expected_rounds_root.require_dataset('round010/76487/nopump/par', **opts)
    expected_rounds_root.require_dataset('round010/76487/nopump/ref', **opts)
    # generate the copied file and make a list of the items present in each file
    reorg.copy_datasets(faulty_raw_data, destination)
    copied_items = []
    destination.visit(lambda name: copied_items.append(name))
    expected_items = []
    expected.visit(lambda name: expected_items.append(name))
    # clean up
    expected.close()
    remove('expected.h5')
    destination.close()
    remove('destination.h5')
    # compare the results
    copied_set = set(copied_items)
    expected_set = set(expected_items)
    assert copied_set == expected_set


def test_creating_wavelengths_top_level_group(organized_clean_data: H5File):
    """Ensures that a wavelength group is created for each wavelength present in the file
    """
    reorg.make_top_level_wavelengths_group(organized_clean_data)
    wav_root = organized_clean_data['wavelengths']
    wavelengths_present = subgroups(wav_root)
    assert set(wavelengths_present) == {'76487', '76715'}
