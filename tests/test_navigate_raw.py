import trcdproc.navigate.raw as nav
from trcdproc.core import H5File


def test_all_signal_dataset_paths_are_found(organized_faulty_data: H5File):
    """Ensures that all dataset paths are found
    """
    dataset_paths_found = {path for path in nav.all_signal_dataset_paths(organized_faulty_data)}
    all_paths = []
    organized_faulty_data.visit(lambda path: all_paths.append(path))
    dataset_paths_present = {'/' + p for p in all_paths
                             if any(sig in p for sig in ['perp', 'par', 'ref'])
                             and 'faulty' not in p}
    assert dataset_paths_found == dataset_paths_present


def test_all_pump_group_paths_are_found(organized_faulty_data: H5File):
    """Ensures that all of the pump/nopump groups are found, and that no faulty groups are picked up
    """
    pump_groups_found = {path for path in nav.pump_group_paths(organized_faulty_data)}
    all_paths = []
    organized_faulty_data.visit(lambda path: all_paths.append(path))
    pump_groups_present = {'/' + p for p in all_paths if p.endswith('pump')}
    assert pump_groups_found == pump_groups_present


def test_all_wavelength_groups_under_rounds_are_found(organized_faulty_data: H5File):
    """Ensures that all of the wavelength groups that are subgroups of rounds are found
    """
    wavelength_groups_found = {path for path in
                               nav.wavelengths_under_rounds_paths(organized_faulty_data)}
    all_paths = []
    organized_faulty_data.visit(lambda path: all_paths.append(path))
    wavelength_groups_present = {'/' + p for p in all_paths
                                 if p.endswith('76487')
                                 or p.endswith('76715')
                                 or p.endswith('76940')}
    assert wavelength_groups_found == wavelength_groups_present
