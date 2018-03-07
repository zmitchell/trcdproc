from os import remove

import h5py
import numpy as np
from pytest import fixture

import trcdproc.reorganize.raw as reorg
from trcdproc.core import H5File, Array


@fixture(scope='function')
def clean_raw_data() -> H5File:
    """Produces a small HDF5 file in the format of the raw experiment data, with no missing data

    Returns:
        An HDF5 file with the following format:
            - File (group)
                - spectrum1 (group)
                    - FMO_176487 (dataset)
                    - FMO_276487 (dataset)
                    - FMO_176715 (dataset)
                    - FMO_276715 (dataset)
                - spectrum2 (group)
                    - (same)
                - spectrum3 (group)
                    - (same)
        Each dataset has 5 columns, each of which holds 50,000 points. The order of the columns is:
            - time
            - perpendicular
            - parallel
            - reference
            - shutter
    """
    file: H5File = h5py.File('clean_raw_data.h5', 'w', libver='latest')
    shutter_open: Array = np.empty(50_000, dtype=np.float64)
    shutter_open.fill(4.8)
    shutter_closed: Array = np.empty(50_000, dtype=np.float64)
    shutter_closed.fill(0.2)
    time: Array = np.arange(0, 4e-4, 8e-9, dtype=np.float64)
    perp: Array = np.random.rand(50_000)
    par: Array = np.random.rand(50_000)
    ref: Array = np.random.rand(50_000)
    open_dset: Array = np.column_stack((time, perp, par, ref, shutter_open))
    closed_dset: Array = np.column_stack((time, perp, par, ref, shutter_closed))
    for spec_name in ['spectrum1', 'spectrum2', 'spectrum3']:
        file.create_group(spec_name)
        for wav_name in ['76487', '76715']:
            open_path: str = f'{spec_name}/FMO_1{wav_name}'
            closed_path: str = f'{spec_name}/FMO_2{wav_name}'
            file[open_path] = open_dset
            file[closed_path] = closed_dset
    yield file
    file.close()
    remove('clean_raw_data.h5')


@fixture(scope='function')
def starts_empty() -> H5File:
    """Produces an empty HDF5 file
    """
    file = h5py.File('empty.h5', 'w', libver='latest')
    yield file
    file.close()
    remove('empty.h5')


@fixture(scope='function')
def has_rounds_groups(clean_raw_data, starts_empty) -> H5File:
    """Produces an HDF5 file containing just the `round00X` groups

    Returns:
        An HDF5 file containing the `File > rounds > round00X` structure
    """
    reorg.create_rounds_from_spectra(clean_raw_data, starts_empty)
    # Note that `starts_empty` is no longer empty at this point
    return starts_empty


@fixture(scope='function')
def faulty_raw_data(clean_raw_data):
    """Returns an HDF5 file where `spectrum1` has an incomplete set of data
    at wavelength 76940 and two spectra with overlapping names
    """
    clean_raw_data['spectrum1/FMO_176940'] = clean_raw_data['spectrum1/FMO_176487'][...]
    clean_raw_data.create_group('spectrum10')
    clean_raw_data['spectrum10/FMO_176487'] = clean_raw_data['spectrum1/FMO_176487'][...]
    clean_raw_data['spectrum10/FMO_276487'] = clean_raw_data['spectrum1/FMO_276487'][...]
    return clean_raw_data


@fixture(scope='function')
def organized_clean_data(clean_raw_data) -> H5File:
    """Produces an HDF5 file with the original data reorganized under the `rounds`
    top level group.
    """
    organized = h5py.File('organized_clean.h5', 'w', libver='latest')
    reorg.create_rounds_from_spectra(clean_raw_data, organized)
    reorg.create_wavelength_groups_for_all_rounds(clean_raw_data, organized)
    reorg.copy_datasets(clean_raw_data, organized)
    yield organized
    organized.close()
    remove('organized_clean.h5')


@fixture(scope='function')
def organized_faulty_data(faulty_raw_data) -> H5File:
    """Produces an HDF5 file with the original data reorganized under the `rounds`
    top level group.
    """
    organized = h5py.File('organized_faulty.h5', 'w', libver='latest')
    reorg.create_rounds_from_spectra(faulty_raw_data, organized)
    reorg.create_wavelength_groups_for_all_rounds(faulty_raw_data, organized)
    reorg.copy_datasets(faulty_raw_data, organized)
    yield organized
    organized.close()
    remove('organized_faulty.h5')
