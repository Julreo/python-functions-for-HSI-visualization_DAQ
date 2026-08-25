import numpy as np
from .io import load_all_images
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
    # Load all images from STEPS subfolder
    images = load_all_images(voltage_dir)
    if images is None:
        print(f"No images loaded from {voltage_dir}")
        return None

    # Unpack calibration data
    wavelengths, pseudo_freqs = calibration_data
    num_wavelengths = len(wavelengths)
    V_size, H_size = images.shape[1], images.shape[2]

    # Initialize HSI cube
    hsi = np.zeros((num_wavelengths, V_size, H_size))

    # Process each pixel
    for i in range(V_size):
        for j in range(H_size):
            # Extract interferogram for this pixel
            interferogram = images[:, i, j]

            # Preprocess: remove DC offset
            processed_interferogram = interferogram - np.mean(interferogram)

            # Apply asymmetric window
            window = create_asymmetric_window(positions)
            processed_interferogram *= window

            # Compute FFT
            amplitude, freq_axis, _ = compute_fft(processed_interferogram, positions)

            # Calibrate spectrum to wavelength axis
            _, calibrated_spectrum = calibrate_spectrum(
                amplitude, freq_axis, wavelengths, pseudo_freqs
            )
            hsi[:, i, j] = calibrated_spectrum

    return hsi