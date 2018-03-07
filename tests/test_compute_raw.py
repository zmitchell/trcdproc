from os import remove

import h5py
import numpy as np
from pytest import fixture, raises

import trcdproc.compute.raw as comp
from trcdproc.core import Array, InputChannel, PumpStatus


@fixture(scope='function')
def data_for_compute_tests():
    """Produces an HDF5 file with the following structure:
        File
            round001
                76487
                    pump
                        time
                        perp
                        par
                        ref
                    nopump
                        time
                        perp
                        par
                        ref
                76715
                    (same structure)
            round002
                (same structure)
        This file is intended to be used for testing the `trcdproc.compute` package, so the data in
        the file is created with known means, noises, etc.
    """
    filename = 'compute.h5'
    file = h5py.File(filename, 'w', libver='latest')
    points = 50_000
    time_data = np.linspace(0, 4e-4, points, dtype=np.float64)
    perp_mean = 0
    par_mean = 1
    ref_mean = 2
    perp_noise = 0.1
    par_noise = 0.2
    ref_noise = 0.3
    perp_data = np.random.normal(perp_mean, perp_noise, points)
    par_data = np.random.normal(par_mean, par_noise, points)
    ref_data = np.random.normal(ref_mean, ref_noise, points)
    rounds_root = file.create_group('rounds')
    for rnd in ['round001', 'round002']:
        for wav in ['76487', '76715']:
            for pump in ['pump', 'nopump']:
                group_path = f'{rnd}/{wav}/{pump}'
                group = rounds_root.require_group(group_path)
                group.create_dataset('time', data=time_data)
                group.create_dataset('perp', data=perp_data)
                group.create_dataset('par', data=par_data)
                group.create_dataset('ref', data=ref_data)
    yield file
    file.close()
    remove(filename)


@fixture(scope='function')
def mean_bounds():
    """The upper and lower bounds for which the computed mean is considered correct
    """
    bounds = {
        'perp': {
            'lower': -0.01,
            'upper': 0.01,
        },
        'par': {
            'lower': 0.99,
            'upper': 1.01,
        },
        'ref': {
            'lower': 1.98,
            'upper': 2.02,
        }
    }
    return bounds


@fixture(scope='function')
def noise_bounds():
    """The upper and lower bounds for which the computed noise is considered correct
    """
    bounds = {
        'perp': {
            'lower': 0.099,
            'upper': 0.101,
        },
        'par': {
            'lower': 0.198,
            'upper': 0.202,
        },
        'ref': {
            'lower': 0.297,
            'upper': 0.303,
        }
    }
    return bounds


def test_channel_heatmap(data_for_compute_tests, mean_bounds):
    """Does a basic test of the heatmap data generator by making a heatmap of
    the means in the pumped reference channel
    """

    def pixel_mean(_: Array, signal: Array):
        """Computes the value of a pixel by computing the mean of the signal
        """
        return signal.mean()

    chan = InputChannel.perp
    pump = PumpStatus.present
    pixels = comp.channel_heatmap(data_for_compute_tests, pixel_mean, chan, pump)
    assert pixels.shape == (2, 2)
    for i in [0, 1]:
        for j in [0, 1]:
            assert pixels[i, j] < mean_bounds['perp']['upper']
            assert pixels[i, j] > mean_bounds['perp']['lower']


def test_store_individual_means(data_for_compute_tests, mean_bounds):
    """Verifies that the correct means are stored for each signal dataset
    """
    comp.store_individual_means(data_for_compute_tests)
    rounds_root = data_for_compute_tests['rounds']
    for rnd in ['round001', 'round002']:
        for wav in ['76487', '76715']:
            for pump in ['pump', 'nopump']:
                with raises(KeyError):  # make sure `mean` wasn't calculated for `time` datasets
                    time_path = f'{rnd}/{wav}/{pump}/time'
                    time_mean = rounds_root[time_path].attrs['mean']
                for sig in ['perp', 'par', 'ref']:
                    dataset_path = f'{rnd}/{wav}/{pump}/{sig}'
                    sig_mean = rounds_root[dataset_path].attrs['mean']
                    assert sig_mean < mean_bounds[sig]['upper']
                    assert sig_mean > mean_bounds[sig]['lower']


def test_store_individual_noises(data_for_compute_tests, noise_bounds):
    """Verifies that the correct noises are stored for each signal dataset
    """
    comp.store_individual_noises(data_for_compute_tests)
    rounds_root = data_for_compute_tests['rounds']
    for rnd in ['round001', 'round002']:
        for wav in ['76487', '76715']:
            for pump in ['pump', 'nopump']:
                with raises(KeyError):  # make sure `noise` wasn't calculated for `time` datasets
                    time_path = f'{rnd}/{wav}/{pump}/time'
                    time_noise = rounds_root[time_path].attrs['noise']
                for sig in ['perp', 'par', 'ref']:
                    dataset_path = f'{rnd}/{wav}/{pump}/{sig}'
                    sig_noise = rounds_root[dataset_path].attrs['noise']
                    assert sig_noise < noise_bounds[sig]['upper']
                    assert sig_noise > noise_bounds[sig]['lower']


def test_store_noise_means(data_for_compute_tests, noise_bounds):
    """Verifies that the correct file-wide means of the noise are stored for each signal channel

    Note:
        The data for a given channel is the same throughout the file, so the mean noise should be
        identical to the individual noises in the channel.
    """
    comp.store_individual_noises(data_for_compute_tests)
    comp.store_noise_means(data_for_compute_tests)
    for sig in ['perp', 'par', 'ref']:
        mean_noise = data_for_compute_tests.attrs[f'{sig}_noise_mean']
        assert mean_noise < noise_bounds[sig]['upper']
        assert mean_noise > noise_bounds[sig]['lower']


def test_store_overall_means(data_for_compute_tests, mean_bounds):
    """Verifies that the correct file-wide means are stored for each signal channel

    Note:
        The data for a given channel is the same throughout the file, so the overall mean should be
        identical to the individual means in the channel.
    """
    comp.store_individual_means(data_for_compute_tests)
    comp.store_overall_means(data_for_compute_tests)
    for sig in ['perp', 'par', 'ref']:
        mean = data_for_compute_tests.attrs[f'{sig}_mean']
        assert mean < mean_bounds[sig]['upper']
        assert mean > mean_bounds[sig]['lower']


def test_store_std_dev_of_means(data_for_compute_tests):
    """Verify that the correct standard deviation of the individual means for each signal channel
    are stored

    Note:
        The individual means for a given channel should be identical,
        so the standard deviation should be very small
    """
    comp.store_individual_means(data_for_compute_tests)
    comp.store_overall_means(data_for_compute_tests)
    comp.store_std_dev_of_means(data_for_compute_tests)
    for sig in ['perp', 'par', 'ref']:
        std_dev = data_for_compute_tests.attrs[f'{sig}_mean_std_dev']
        assert std_dev < 1e-4


def test_store_std_dev_of_noises(data_for_compute_tests):
    """Verifies that the correct standard deviation of the individual noises for each signal channel
    are stored

    Note:
        The individual noises for a given channel should be very similar,
        so the standard deviation should be very small
    """
    comp.store_individual_noises(data_for_compute_tests)
    comp.store_noise_means(data_for_compute_tests)
    comp.store_std_dev_of_noises(data_for_compute_tests)
    for sig in ['perp', 'par', 'ref']:
        std_dev = data_for_compute_tests.attrs[f'{sig}_noise_std_dev']
        assert std_dev < 1e-2
