from os import remove

import h5py
import numpy as np
from pytest import fixture, raises

import trcdproc.compute.absorp as compute


@fixture(scope='function')
def delta_a_clean_input_data():
    """Constructs an HDF5 file with specific values for the different signals for the sake of
    testing the correctness of the computed change in absorption

    The change in absorption (dA) is calculated by with the following equation:
        dA = -log10(
            (probe with pump) / (ref with pump)
            /
            (probe without pump) / (ref without pump)
        )

    The test should reveal whether the correct value is being produced, and whether the probe
    signals are being divided by their corresponding reference signals. For this reason the
    following values have been chosen for the various signals:
        probe with pump = 300 (both perp and par)
        reference with pump = 3
        probe without pump = 20 (both per and par)
        reference without pump = 2
    For these values the expected change in absorption is -1.
    """
    filename = 'delta_a_inputs.h5'
    file = h5py.File(filename, 'w', libver='latest')
    group = file.require_group('/rounds/round000/76487')
    pump = group.create_group('pump')
    nopump = group.create_group('nopump')
    # create the data for the dA calculations
    probe_with_pump_value = 300
    ref_with_pump_value = 3
    probe_without_pump_value = 20
    ref_without_pump_value = 2
    time_data = np.arange(0, 101, 1, dtype=np.float64)
    points = 100
    probe_with_pump = np.empty(points, dtype=np.float64)
    probe_with_pump.fill(probe_with_pump_value)
    probe_without_pump = np.empty(points, dtype=np.float64)
    probe_without_pump.fill(probe_without_pump_value)
    ref_with_pump = np.empty(points, dtype=np.float64)
    ref_with_pump.fill(ref_with_pump_value)
    ref_without_pump = np.empty(points, dtype=np.float64)
    ref_without_pump.fill(ref_without_pump_value)
    pump.create_dataset('time', data=time_data, dtype=np.float64)
    pump.create_dataset('perp', data=probe_with_pump, dtype=np.float64)
    pump.create_dataset('par', data=probe_with_pump, dtype=np.float64)
    pump.create_dataset('ref', data=ref_with_pump, dtype=np.float64)
    nopump.create_dataset('time', data=time_data, dtype=np.float64)
    nopump.create_dataset('perp', data=probe_without_pump, dtype=np.float64)
    nopump.create_dataset('par', data=probe_without_pump, dtype=np.float64)
    nopump.create_dataset('ref', data=ref_without_pump, dtype=np.float64)
    yield file
    # clean up
    file.close()
    remove(filename)


@fixture(scope='function')
def data_for_baseline_adjustment():
    """Produces an HDF5 file with constant data in each channel for testing whether the baseline
    adjustment works properly.
    """
    filename = 'baseline.h5'
    file = h5py.File(filename, 'w', libver='latest')
    wav_group = file.require_group('/rounds/round000/76487')
    # generate the data
    time_data = np.ones(50_000, dtype=np.float64)
    perp_data = np.ones(50_000, dtype=np.float64)
    par_data = np.empty(50_000, dtype=np.float64)
    par_data.fill(2.0)
    wav_group.create_dataset('time', data=time_data, dtype=np.float64)
    wav_group.create_dataset('perp', data=perp_data, dtype=np.float64)
    wav_group.create_dataset('par', data=par_data, dtype=np.float64)
    yield file
    # clean up
    file.close()
    remove(filename)


@fixture(scope='function')
def data_for_collapse():
    filename = 'collapse.h5'
    file = h5py.File(filename, 'w', libver='latest')
    time_data = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)
    abs_data = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.float64)
    file.create_dataset('time', data=time_data, dtype=np.float64)
    file.create_dataset('perp', data=abs_data, dtype=np.float64)
    yield file
    # clean up
    file.close()
    remove(filename)


def test_compute_single_clean_delta_a(delta_a_clean_input_data, starts_empty):
    """Verifies that the expected change in absorption is calculated with clean data
    i.e. data that does not produce NaN when the logarithm is computed
    """
    wavelength_group = delta_a_clean_input_data['rounds/round000/76487']
    compute.compute_single_delta_a(wavelength_group, starts_empty)
    perp_delta_a = starts_empty['perp'][0]
    par_delta_a = starts_empty['par'][0]
    assert perp_delta_a < -0.99
    assert perp_delta_a > -1.01
    assert par_delta_a < -0.99
    assert par_delta_a > -1.01


def test_compute_single_delta_a_with_nans(delta_a_clean_input_data, starts_empty):
    """Verifies that no NaNs make it into the output when the logarithm is taken of
    a negative number.
    """
    wavelength_group = delta_a_clean_input_data['rounds/round000/76487']
    wavelength_group['pump/perp'][...] *= -1
    compute.compute_single_delta_a(wavelength_group, starts_empty)
    should_be_zero = starts_empty['perp'][0]
    assert should_be_zero == 0


def test_compute_all_delta_a(delta_a_clean_input_data, starts_empty):
    """Verifies that all of the data in the original file makes it into the new file
    """
    rounds_root = delta_a_clean_input_data['rounds']
    rounds_root.copy(rounds_root['round000/76487'], rounds_root['round000'], name='76715')
    rounds_root.copy(rounds_root['round000'], rounds_root, name='round001')
    compute.compute_all_delta_a(delta_a_clean_input_data, starts_empty)
    for rnd in ['round000', 'round001']:
        for wav in ['76487', '76715']:
            for sig in ['perp', 'par']:
                path = f'rounds/{rnd}/{wav}/{sig}'
                delta_a = starts_empty[path][0]
                assert delta_a < -0.99
                assert delta_a > -1.01


def test_compute_only_good_delta_a(delta_a_clean_input_data, starts_empty):
    """Verifies that the change in absorption is calculated for wavelength groups that have a value
    of `False` for the `isbad` attribute.
    """
    rounds_root = delta_a_clean_input_data['rounds']
    rounds_root.copy(rounds_root['round000/76487'], rounds_root['round000'], name='76715')
    rounds_root['round000/76487'].attrs['isbad'] = True
    rounds_root['round000/76715'].attrs['isbad'] = False
    compute.compute_only_good_delta_a(delta_a_clean_input_data, starts_empty)
    with raises(KeyError):
        does_not_exist = starts_empty['rounds/round000/76487']  # noqa
    delta_a = starts_empty['rounds/round000/76715/perp'][0]
    assert delta_a < -0.99
    assert delta_a > -1.01


def test_baseline_adjustment(data_for_baseline_adjustment, starts_empty):
    """Verifies that the baseline adjustment is done properly for each channel individually.
    """
    compute.adjust_baseline(data_for_baseline_adjustment, starts_empty)
    for chan in ['time', 'perp', 'par']:
        path = f'rounds/round000/76487/{chan}'
        if chan == 'time':
            assert starts_empty[path][0] > 0.99  # time shouldn't be adjusted
            assert starts_empty[path][0] < 1.01
        else:
            assert starts_empty[path][0] > -0.01
            assert starts_empty[path][0] < 0.01


def test_collapse_single(data_for_collapse):
    """Verifies that the collapse algorithm works as intended.
    """
    time_data = data_for_collapse['time']
    abs_data = data_for_collapse['perp']
    collapsed_time, collapsed_abs = compute.collapse(time_data, abs_data, 3)
    for a in collapsed_abs:
        assert a > 1.99
        assert a < 2.01
    assert collapsed_time[0] > 1.99
    assert collapsed_time[0] < 2.01
    assert collapsed_time[1] > 4.99
    assert collapsed_time[1] < 5.01
