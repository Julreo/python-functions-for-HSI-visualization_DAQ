"""Regression tests for hsi_analysis.core (run with: python -m unittest discover)."""
import unittest

import numpy as np

from hsi_analysis.core import (
    create_asymmetric_window,
    compute_fft,
    calibrate_spectrum,
    compute_hsi_cube,
)


class TestCreateAsymmetricWindow(unittest.TestCase):
    def test_shape_and_edges(self):
        positions = np.linspace(-5, 10, 16)
        window = create_asymmetric_window(positions)
        zpd_index = np.argmin(np.abs(positions))

        self.assertEqual(window.shape, positions.shape)
        self.assertAlmostEqual(window[zpd_index], 1.0, places=10)
        self.assertAlmostEqual(window[0], 0.0, places=10)
        self.assertAlmostEqual(window[-1], 0.0, places=10)


class TestComputeFft(unittest.TestCase):
    def test_output_shapes_and_freq_axis_sorted(self):
        positions = np.linspace(-3, 6, 10)
        interferogram = np.sin(positions)
        amplitude, freq_axis, window = compute_fft(interferogram, positions)

        pad_length = int(2 ** np.ceil(np.log2(len(interferogram))))
        self.assertEqual(amplitude.shape, (pad_length,))
        self.assertEqual(freq_axis.shape, (pad_length,))
        self.assertTrue(np.all(np.diff(freq_axis) > 0))
        self.assertEqual(window.shape, positions.shape)


class TestComputeHsiCube(unittest.TestCase):
    def test_matches_per_pixel_loop(self):
        """compute_hsi_cube must be numerically equivalent to a naive
        per-pixel compute_fft/calibrate_spectrum loop (guards against the
        vectorization introducing regressions)."""
        rng = np.random.default_rng(0)
        n_frames, height, width = 12, 3, 4
        images = rng.normal(size=(n_frames, height, width))
        positions = np.linspace(-4, 8, n_frames)

        # Calibration axis kept inside the FFT's frequency range so no clamping occurs.
        _, freq_axis, _ = compute_fft(images[:, 0, 0], positions)
        pseudo_freqs = np.linspace(freq_axis.min() * 0.5, freq_axis.max() * 0.5, 5)
        wavelengths = np.linspace(500, 900, len(pseudo_freqs))

        expected = np.empty((len(wavelengths), height, width))
        for i in range(height):
            for j in range(width):
                amplitude, freq_axis, _ = compute_fft(images[:, i, j], positions)
                _, expected[:, i, j] = calibrate_spectrum(amplitude, freq_axis, wavelengths, pseudo_freqs)

        actual = compute_hsi_cube(images, positions, wavelengths, pseudo_freqs)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_output_shape(self):
        images = np.zeros((8, 2, 2))
        positions = np.linspace(-1, 1, 8)
        wavelengths = np.array([600.0, 700.0])
        pseudo_freqs = np.array([0.0, 0.1])

        hsi = compute_hsi_cube(images, positions, wavelengths, pseudo_freqs)
        self.assertEqual(hsi.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
