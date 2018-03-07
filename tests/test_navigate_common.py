import trcdproc.navigate.common as nav
from trcdproc.core import H5File


def test_wavelength_set_is_complete(organized_faulty_data: H5File):
    """Ensures that the set of wavelengths present in a file is collected correctly
    """
    wavelengths_present = nav.wavelength_set(organized_faulty_data)
    assert wavelengths_present == {'76487', '76715', '76940'}


def test_all_dataset_paths_below_are_collected(organized_faulty_data: H5File):
    """Ensures that all dataset paths are collected
    """
    dataset_paths_found = set(nav.dataset_paths_below(organized_faulty_data, 'rounds'))
    all_paths = []
    organized_faulty_data.visit(lambda path: all_paths.append(path))
    dataset_paths_present = set([p for p in all_paths
                                 if any(sig in p for sig in ['time', 'perp', 'par', 'ref'])
                                 ])
    assert dataset_paths_found == dataset_paths_present
