import numpy as np

import trcdproc.compute.common as compute


def test_noise_is_reasonable():
    """Verifies that the method used to calculate the noise of a time-varying signal
    produces reasonable results i.e. +/- 1%
    """
    points = 1_000
    x_data = np.linspace(0, 100, points, dtype=np.float64)  # evenly spaced points between 0 and 100
    period = 10
    angular_frequency = 2 * np.pi / period
    signal = np.sin(angular_frequency * x_data)  # a sine wave with a period of 10
    noise_mean = 0
    noise_std_dev = 1.0
    simulated_noise = np.random.normal(noise_mean, noise_std_dev, points)  # Gaussian noise
    noisy_signal = signal + simulated_noise
    calculated_noise = compute.noise(noisy_signal)
    assert calculated_noise < 1.01 * noise_std_dev
    assert calculated_noise > 0.99 * noise_std_dev
