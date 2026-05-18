import numpy as np
from scipy.fft import fft, fftshift

def create_asymmetric_window(positions):
    """Create asymmetric apodization window centered on ZPD."""
    zpd_index = np.argmin(np.abs(positions))
    n = len(positions)
    window = np.zeros(n)

    if zpd_index > 0:
        left_length = zpd_index + 1
        left_window = np.linspace(0, 1, left_length)
        window[:zpd_index+1] = left_window

    if zpd_index < n-1:
        right_length = n - zpd_index
        right_window = np.linspace(1, 0, right_length)
        window[zpd_index:] = right_window

    window = np.sin(np.pi/2 * window)**2
    return window

def compute_fft(interferogram, positions):
    """Compute FFT with asymmetric apodization centered on ZPD."""
    interferogram = interferogram - np.mean(interferogram)
    window = create_asymmetric_window(positions)
    windowed_interferogram = interferogram * window

    n = len(interferogram)
    pad_length = int(2 ** np.ceil(np.log2(n)))
    padded = np.pad(windowed_interferogram, (0, pad_length - n), 'constant')

    fft_result = fftshift(fft(padded))
    delta_pos = np.mean(np.diff(positions))
    freq_axis = fftshift(np.fft.fftfreq(pad_length, d=delta_pos))

    return np.abs(fft_result), freq_axis, window

def calibrate_spectrum(amplitude, freq_axis, wavelengths, pseudo_freqs):
    """Calibrate amplitude spectrum to wavelengths."""
    return wavelengths, np.interp(pseudo_freqs, freq_axis, amplitude)