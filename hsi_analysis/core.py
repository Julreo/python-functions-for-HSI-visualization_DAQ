import numpy as np
from .io import load_all_images
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d

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

def compute_hsi_cube(images, positions, wavelengths, pseudo_freqs):
    """
    Compute an HSI cube for a full image stack in one vectorized pass
    (equivalent to calling compute_fft/calibrate_spectrum per-pixel, but
    without the Python-level pixel loop).

    Args:
        images (np.ndarray): Interferogram stack (frames × height × width)
        positions (np.ndarray): Array of position values (deltax)
        wavelengths (np.ndarray): Calibrated wavelength axis
        pseudo_freqs (np.ndarray): Pseudo-frequencies matching `wavelengths`

    Returns:
        np.ndarray: 3D HSI cube (wavelengths × height × width)
    """
    n = images.shape[0]
    demeaned = images - images.mean(axis=0, keepdims=True)
    window = create_asymmetric_window(positions)
    windowed = demeaned * window[:, None, None]

    pad_length = int(2 ** np.ceil(np.log2(n)))
    padded = np.pad(windowed, ((0, pad_length - n), (0, 0), (0, 0)), 'constant')

    amplitude = np.abs(fftshift(fft(padded, axis=0), axes=0))
    delta_pos = np.mean(np.diff(positions))
    freq_axis = fftshift(np.fft.fftfreq(pad_length, d=delta_pos))

    # np.interp clamps out-of-range values to the boundary amplitude; replicate that here.
    interpolator = interp1d(
        freq_axis, amplitude, axis=0, kind='linear',
        bounds_error=False, fill_value=(amplitude[0], amplitude[-1]),
    )
    return interpolator(pseudo_freqs).astype(np.float32)

def compute_hsi_for_voltage(voltage_dir, positions, calibration_data):
    """
    Compute HSI cube for a single voltage folder.

    Args:
        voltage_dir (str): Path to the voltage folder (e.g., .../Vds=0.00V/Vgs=-0.40V/STEPS/)
        positions (np.ndarray): Array of position values (deltax)
        calibration_data (tuple): (wavelengths, pseudo_freqs) from calibration file

    Returns:
        np.ndarray: 3D HSI cube (wavelengths × height × width)
    """
    images = load_all_images(voltage_dir)
    if images is None:
        print(f"No images loaded from {voltage_dir}")
        return None

    wavelengths, pseudo_freqs = calibration_data
    return compute_hsi_cube(images, positions, wavelengths, pseudo_freqs)