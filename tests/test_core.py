from trcdproc.core import subgroups, datasets, count_signal_datasets


def test_conftest(clean_raw_data):
    """Ensures that the test data is generated correctly

    Args:
        clean_raw_data (H5File): An HDF5 file containing clean test data
    """
    assert clean_raw_data is not None


def test_subgroups(clean_raw_data):
    """Ensures that the correct subgroup names are reported

    Args:
        clean_raw_data (H5File): An HDF5 file containing clean test data
    """
    subgroup_names = subgroups(clean_raw_data)
    assert subgroup_names == ['spectrum1', 'spectrum2', 'spectrum3']


def test_datasets(clean_raw_data):
    """Ensures that the correct dataset names are reported

    Args:
        clean_raw_data (H5File): An HDF5 file containing clean test data
    """
    dataset_names = datasets(clean_raw_data['spectrum1'])
    assert dataset_names == ['FMO_176487', 'FMO_276487', 'FMO_176715', 'FMO_276715']


def test_count_signal_datasets(organized_clean_data):
    """Ensures that the number of datasets is properly counted recursively
    """
    dataset_paths = []
    organized_clean_data.visit(lambda path: dataset_paths.append(path))
    signal_paths = [p for p in dataset_paths if any(sig in p for sig in ['perp', 'par', 'ref'])]
    expected_count = len(signal_paths)
    actual_count = count_signal_datasets(organized_clean_data)
    assert actual_count == expected_count
