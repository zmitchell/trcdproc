from trcdproc.core import subgroups, H5File
from trcdproc.navigate.common import wavelength_set


def copy_rounds_structure_for_delta_a(old_file: H5File, new_file: H5File) -> None:
    """Copies the File/roundX/WWWWW group structure into a new, empty file for
    storing dA data in at a later time

    Args:
        old_file (H5File): The file whose structure will be copied
        new_file (H5File): An empty file that will have a group structure copied into it
    """
    rounds_root = old_file['rounds']
    for rnd in subgroups(rounds_root):
        for wav in subgroups(rounds_root[rnd]):
            path = f'rounds/{rnd}/{wav}'
            new_file.require_group(path)
    return


def make_delta_a_wavelength_groups(src_file: H5File) -> None:
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

    Note:
        Not covered by tests yet

    Args:
        src_file (H5File): A file that has already had its data reorganized into rounds
    """
    src_file.create_group('wavelengths')
    wav_root = src_file['wavelengths']
    all_wavelengths = wavelength_set(src_file)
    for wav_name in all_wavelengths:
        wav_root.create_group(wav_name)
    rounds_root = src_file['rounds']
    for rnd_name in subgroups(rounds_root):
        for wav_name in subgroups(rounds_root[rnd_name]):
            wav_root[wav_name].create_group(rnd_name)
            old_time_path = f'{rnd_name}/{wav_name}/time'
            old_perp_path = f'{rnd_name}/{wav_name}/perp'
            old_par_path = f'{rnd_name}/{wav_name}/par'
            new_time_path = f'{wav_name}/{rnd_name}/time'
            new_perp_path = f'{wav_name}/{rnd_name}/perp'
            new_par_path = f'{wav_name}/{rnd_name}/par'
            wav_root[new_time_path] = rounds_root[old_time_path]
            wav_root[new_perp_path] = rounds_root[old_perp_path]
            wav_root[new_par_path] = rounds_root[old_par_path]
    return
