import re
from typing import Dict, List, Tuple

import h5py
import numpy as np
from logzero import logger

from trcdproc.core import (
    H5File,
    Dataset,
    Group,
    subgroups,
    datasets,
)
from trcdproc.navigate.common import wavelength_set
from trcdproc.reorganize.common import recursive_copy


def is_with_pump(dataset: Dataset) -> bool:
    """Determines whether the dataset was collected with or without the pump present

    Args:
        dataset (Dataset): The dataset whose pump status you want to query

    Returns:
        True if the pump is present, False if it is absent
    """
    shutter_mean = dataset[:, 4].mean(axis=0)
    if shutter_mean < 2.5:
        logger.debug(f'Dataset {dataset.name} is with pump')
        return True
    else:
        logger.debug(f'Dataset {dataset.name} is not with pump')
        return False


def create_rounds_from_spectra(old_file: H5File, new_file: H5File) -> None:
    """Creates a group 'roundX' for each top level group in the original file named 'spectrumX'

    Args:
        old_file (H5File): The original file with the experimental data
        new_file (H5File): The new, empty file in which the data will be organized
    """
    spec_number_regex = re.compile(r'spectrum(\d+)')
    new_file.create_group('rounds')
    for s in subgroups(old_file):
        match = spec_number_regex.search(s)
        if match is None:
            continue
        spec_number = match.group(1)
        new_file['rounds'].create_group(f'round{spec_number:0>3s}')
    return


def create_wavelength_groups_for_all_rounds(old_file: H5File, new_file: H5File) -> None:
    """Creates the subgroups in the new file for each wavelength present in a round

    Args:
        old_file (H5File): The original file with the experimental data
        new_file (H5File): The new file in which the data will be organized
    """
    rounds_root = new_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    spectra = sorted(subgroups(old_file))
    spectra = [s for s in spectra if s != 'experiment_parameters']
    spectrum_map = make_spectrum_map(spectra)
    spectra_for_zip = sorted(spectrum_map.keys())
    for spec, rnd in zip(spectra_for_zip, rounds):
        real_spec_name = spectrum_map[spec]
        create_wavelength_groups_for_one_round(old_file[real_spec_name], rounds_root[rnd])
    return


def make_spectrum_map(spectra: List[str]) -> Dict[str, str]:
    """Creates a dictionary mapping a 'spectrum00X' name to the real 'spectrumX' name.

    The reason this is necessary is that the rounds are stored as 'round00X', but the spectra are
    stored as 'spectrumX'. The two different naming schemes are sorted differently:
        spectrum1
        spectrum10
        spectrum11
        etc
    vs.
        round001
        round002
        etc
    This gives unexpected results when iterating through 'zip(spectra, rounds)'.

    Args:
        spectra (List[str]): The list of 'spectrumX' names

    Returns:
        The mapping of 'spectrum00X' to 'spectrumX'
    """
    spec_number_regex = re.compile(r'spectrum(\d+)')
    spectra_map: Dict[str, str] = dict()
    for spec in spectra:
        match = spec_number_regex.search(spec)
        if match is None:
            continue
        spec_number = match.group(1)
        spectra_map[f'spectrum{spec_number:0>3s}'] = spec
    return spectra_map


def create_wavelength_groups_for_one_round(old_parent_group: Group, new_parent_group: Group) -> None:  # noqa
    """Creates subgroups under the new parent group for each wavelength present in the old parent group

    Args:
        old_parent_group (Group): The group under which the old experimental datasets are stored
        new_parent_group (Group): The group under which a new group will be created for each
            wavelength in the old datasets
    """
    dataset_names = sorted(datasets(old_parent_group))
    wavelength_regex = re.compile(r'FMO_[12](\d+)$')
    wavelength_matches = [wavelength_regex.search(n) for n in dataset_names]
    wavelengths = [m.group(1) for m in wavelength_matches if m is not None]
    deduped_wavelengths = set(wavelengths)
    for w in deduped_wavelengths:
        new_parent_group.create_group(w)
        new_parent_group[w].attrs['wavelength'] = w
    return


def copy_datasets(old_file: H5File, new_file: H5File) -> None:
    """Walks the top level groups in the old file to copy datasets to the new file,
    dropping the last column in each dataset and renaming the dataset to either "pump"
    or "nopump"

    Args:
        old_file (H5File): The original file with the experimental data
        new_file (H5File): The new file in which the data will be organized
    """
    rounds_root = new_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    spectra = [s for s in subgroups(old_file) if s != 'experiment_parameters']
    spectrum_map = make_spectrum_map(spectra)
    spectra_for_zip = sorted(spectrum_map.keys())
    for spec, rnd in zip(spectra_for_zip, rounds):
        real_spec_name = spectrum_map[spec]
        dataset_names = datasets(old_file[real_spec_name])
        pairs = sorted(pair_dataset_names(dataset_names))
        for wav, *dset_names in pairs:
            if len(dset_names) < 2:
                dset_old_path = f'/{real_spec_name}/{dset_names[0]}'
                dset = old_file[dset_old_path]
                dset_new_path = f'/rounds/{rnd}/{wav}/faulty1'
                new_file.create_group(dset_new_path)
                logger.debug(f'Copying dataset {dset_old_path} to {dset_new_path}')
                split_and_store_old_dataset(new_file[dset_new_path], dset)
                continue
            dset1_name = dset_names[0]
            dset2_name = dset_names[1]
            dset1_old_path = f'/{real_spec_name}/{dset1_name}'
            dset2_old_path = f'/{real_spec_name}/{dset2_name}'
            dset1 = old_file[dset1_old_path]
            dset2 = old_file[dset2_old_path]
            dset1_new_path, dset2_new_path = round_path_for_old_dataset(dset1, dset2, rnd, wav)
            new_file.create_group(dset1_new_path)
            new_file.create_group(dset2_new_path)
            new_group1 = new_file[dset1_new_path]
            new_group2 = new_file[dset2_new_path]
            logger.debug(f'Copying dataset {dset1_old_path} to {dset1_new_path}')
            split_and_store_old_dataset(new_group1, dset1)
            logger.debug(f'Copying dataset {dset2_old_path} to {dset2_new_path}')
            split_and_store_old_dataset(new_group2, dset2)
    return


def round_path_for_old_dataset(dset1: Dataset, dset2: Dataset, rnd: str, wav: str) -> Tuple[str, str]:  # noqa
    """Determines the path for the group which will store the data extracted from the dataset

    Args:
        dset1 (Dataset): The first dataset in the pump/nopump pair
        dset2 (Dataset): The second dataset in the pump/nopump pair
        rnd (str): The round in which the new group will be stored i.e. 'round0' or 'round5'
        wav (str): The wavelength at which the data was collected i.e. '76487'

    Returns:
        new_paths (Tuple[str, str]): The paths to the new groups
    """
    dset1_has_pump = is_with_pump(dset1)
    dset2_has_pump = is_with_pump(dset2)
    if dset1_has_pump and dset2_has_pump:
        dset1_new_path = f'/rounds/{rnd}/{wav}/faulty1'
        dset2_new_path = f'/rounds/{rnd}/{wav}/faulty2'
    elif dset1_has_pump:
        dset1_new_path = f'/rounds/{rnd}/{wav}/pump'
        dset2_new_path = f'/rounds/{rnd}/{wav}/nopump'
    elif dset2_has_pump:
        dset1_new_path = f'/rounds/{rnd}/{wav}/nopump'
        dset2_new_path = f'/rounds/{rnd}/{wav}/pump'
    else:
        dset1_new_path = f'/rounds/{rnd}/{wav}/faulty1'
        dset2_new_path = f'/rounds/{rnd}/{wav}/faulty2'
    return dset1_new_path, dset2_new_path


def split_and_store_old_dataset(new_group: Group, old_dataset: Dataset) -> None:
    """Splits the multicolumn array from the old dataset into single-column datasets

    Args:
        new_group (Group): The group in which to store the new datasets
        old_dataset (Dataset): The old data to split
    """
    new_group.create_dataset('time', data=old_dataset[:, 0], dtype=np.float32)
    new_group.create_dataset('perp', data=old_dataset[:, 1], dtype=np.float32)
    new_group.create_dataset('par', data=old_dataset[:, 2], dtype=np.float32)
    new_group.create_dataset('ref', data=old_dataset[:, 3], dtype=np.float32)
    return


def pair_dataset_names(names: List[str]) -> List[Tuple[str, str, str]]:
    """Returns a list of pairs of datasets and at the same wavelength along with the wavelength

    Args:
        names (List[str]): A list of the dataset names to process

    Returns:
        The pairs of dataset names at each wavelength in
        the format (wavelength, name, name)
    """
    wavelength_regex = re.compile(r'^FMO_[12](\d+)')
    wavelengths = [wavelength_regex.search(n).group(1) for n in names
                   if wavelength_regex.search(n) is not None]
    wavelengths = set(wavelengths)
    pairs = []
    for w in wavelengths:
        selection = [n for n in names if w in n]
        pairs.append((w, *selection))
    return pairs


def reorganize_rounds(old_file: H5File, new_file: H5File) -> None:
    """Converts experiment data from the old structure to the new structure, discarding the
    shutter column in the process

    The old format is as follows:
    File
        spectrum<X>
            FMO_1<wavelength1> <- 5 column dataset
            FMO_2<wavelength1>
            FMO_1<wavelength2>
            FMO_2<wavelength2>
            etc

    The new format is as follows:
    File
        rounds
            round<X>
                <wavelength1>
                    pump
                        time <- 1 column dataset
                        perp
                        par
                        ref
                    nopump
                        time
                        perp
                        par
                        ref
                <wavelength2>
                    etc

    Note:
        The name 'spectrum' has been changed to 'round' since the data contained in each 'spectrum'
        is not actually a spectrum, but rather one cycle through all of the wavelengths.

    Note:
        The shutter column is discarded because it is trivial to determine whether the pump was
        present and store that information in the name of the enclosing group

    Args:
        old_file (H5File): The file containing the original experiment data
        new_file (H5File): The file to store the reorganized rounds in
    """
    logger.info(f'Creating the rounds structure in {new_file.filename}')
    create_rounds_from_spectra(old_file, new_file)
    create_wavelength_groups_for_all_rounds(old_file, new_file)
    logger.info(f'Copying datasets from {old_file.filename} to {new_file.filename}')
    copy_datasets(old_file, new_file)
    logger.info(f'Done reorganizing rounds into {new_file.filename}')
    return


def make_top_level_wavelengths_group(organized_file: H5File) -> None:
    """Takes the newly reorganized data and creates a new top level group that divides the data
    by wavelength rather than round.

    The structure of the new wavelength groups will be as follows:
    File
        wavelengths
            WWWWW
                roundX
                    pump
                    nopump
    Note that these new groups and datasets are simply hard links, not new copies of the data,
    so no space penalty is incurred by adding a new organizational structure like this.

    Args:
        organized_file (H5File): A file that has already had its data reorganized into rounds
    """
    logger.info(f'Assembling a top level \'wavelengths\' group in {organized_file.filename}')
    organized_file.create_group('wavelengths')
    wav_root = organized_file['wavelengths']
    all_wavelengths = wavelength_set(organized_file)
    for wav_name in all_wavelengths:
        wav_root.create_group(wav_name)
    rounds_root = organized_file['rounds']
    for rnd_name in subgroups(rounds_root):
        for wav_name in subgroups(rounds_root[rnd_name]):
            wav_root[wav_name].create_group(rnd_name)
            old_wav_path = f'{rnd_name}/{wav_name}'
            for pump_name in subgroups(rounds_root[old_wav_path]):
                old_pump_path = f'{rnd_name}/{wav_name}/{pump_name}'
                new_pump_path = f'{wav_name}/{rnd_name}/{pump_name}'
                wav_root[new_pump_path] = rounds_root[old_pump_path]
    logger.info(f'Done creating \'wavelengths\' group')
    return


def restructure_experiment_data(old_file: H5File, new_file: H5File) -> None:
    """Reorganizes the structure of the original experiment into a more intuitive
    group structure, including top level groups organized by rounds and wavelengths

    Args:
        old_file (H5File): the file with the original experiment data
        new_file (H5File): an empty file into which the reorganized data will be stored
    """
    logger.info(f'Restructuring the data in {old_file.filename} into {new_file.filename}')
    reorganize_rounds(old_file, new_file)
    make_top_level_wavelengths_group(new_file)
    logger.info(f'Done restructuring {new_file.filename}')
    return


def print_missing(filenames: List[str]) -> None:
    """Walks the file looking for wavelength groups that don't have any pump or faulty subgroups

    Args:
        filenames (List[str]): The list of filenames to inspect
    """
    missing = []
    for filename in filenames:
        data = h5py.File(filename, 'r')
        rounds_root = data['rounds']
        for rnd_name in subgroups(rounds_root):
            for wav_name in subgroups(rounds_root[rnd_name]):
                wav_group = rounds_root[rnd_name + '/' + wav_name]
                contents = subgroups(wav_group)
                if len(contents) == 0:
                    path = data.filename[0:-3] + ': ' + wav_group.name
                    missing.append(path)
        data.close()
    print(f'Number of missing files: {len(missing)}')
    for path in missing:
        print(path)
    return


def combine_files(filenames: List[str], joined_name: str) -> None:
    """Combines multiple small files into one large file with the rounds renamed

    Args:
        filenames (List[str]): The list of filenames to combine
        joined_name (str): The name of the file to store the merged data in i.e. 'joined.h5'

    Note:
        Not covered by tests yet

    Warning:
        This function recombines files that have already been restructured. Do not use this
        on raw data files.
    """
    logger.info(f'Combining {filenames} into {joined_name}')
    new_file = h5py.File(joined_name, 'w', libver='latest')
    new_file.create_group('rounds')
    new_rounds_root = new_file['rounds']
    rnd_counter = 0
    for filename in filenames:
        logger.debug(f'Copying {filename} into {joined_name}')
        old_file = h5py.File(filename, 'r', libver='latest')
        old_rounds_root = old_file['rounds']
        for old_rnd_name in sorted(subgroups(old_rounds_root)):
            new_rnd_name = f'round{rnd_counter:0>3d}'
            new_rounds_root.create_group(new_rnd_name)
            recursive_copy(old_rounds_root[old_rnd_name], new_rounds_root[new_rnd_name])
            rnd_counter += 1
        old_file.close()
    new_file.close()
    return
