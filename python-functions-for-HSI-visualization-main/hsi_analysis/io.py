import numpy as np
import os
import glob

def load_all_images(folder_path):
    """Load all images from a folder (16-bit data)."""
    bin_files = sorted(glob.glob(os.path.join(folder_path, "*.bin")))
    if not bin_files:
        print(f"No .bin files found in {folder_path}")
        return None

    # Read first file as int16 to get dimensions
    first_img = np.fromfile(bin_files[0], dtype=np.int16)
    V_size, H_size = int(first_img[0]), int(first_img[2])
    pixel_data = first_img[4:].astype(np.float32)  # Convert pixel data to float32

    # Reshape first image
    first_img_reshaped = pixel_data.reshape((V_size, H_size))

    # Pre-allocate array for all images
    N = len(bin_files)
    images = np.empty((N, V_size, H_size), dtype=np.float32)
    images[0] = first_img_reshaped

    # Load remaining images
    for i in range(1, N):
        img = np.fromfile(bin_files[i], dtype=np.int16)[4:].astype(np.float32)
        images[i] = img.reshape((V_size, H_size))

    return images

def load_positions_ss(folder_path):
    """Load positions.txt from Vgs folder (now in Vds/Vgs/STEPS/)."""
    # Try parent folder first (Vgs folder)
    positions_path = os.path.join(folder_path, "positions.txt")
    if os.path.exists(positions_path):
        positions = np.loadtxt(positions_path)
        return positions

    # Fallback to STEPS subfolder (if needed)
    steps_path = os.path.join(folder_path, "STEPS", "positions.txt")
    if os.path.exists(steps_path):
        positions = np.loadtxt(steps_path)
        return positions

    print(f"No positions.txt found in {folder_path} or {os.path.join(folder_path, 'STEPS')}")
    return None
   
def load_positions_dyn(data_dir):
    """Load positions from dataname_POS.txt."""
    positions_file = glob.glob(os.path.join(data_dir, '*POS.txt'))[0]
    try:
        return np.loadtxt(positions_file, dtype=np.float64, delimiter='\t')
    except Exception as e:
        print(f"Error loading positions from {positions_file}: {str(e)}")
        return None
   
def load_calibration_ss(data_dir):
    """Load calibration data from {data_dir}/calibration.txt."""
    calibration_file = os.path.join(data_dir, 'calibration.txt')

    if not os.path.exists(calibration_file):
        print(f"Calibration file not found at {calibration_file}")
        return None, None

    try:
        calibration_data = np.loadtxt(calibration_file)
        wavelengths = calibration_data[0, :]  # First row: wavelengths
        pseudo_freqs = calibration_data[1, :]  # Second row: pseudo-frequencies
        return wavelengths, pseudo_freqs
    except Exception as e:
        print(f"Error loading calibration from {calibration_file}: {str(e)}")
        return None, None

def load_calibration_dyn(data_dir):
    """Load calibration from calibration.txt in data directory."""
    calibration_file = os.path.join(data_dir, 'calibration.txt')
    try:
        data = np.loadtxt(calibration_file, dtype=np.float64, delimiter='\t')
        return data[0, :], data[1, :]
    except Exception as e:
        print(f"Error loading calibration from {calibration_file}: {str(e)}")
        return None, None

def save_hsi_dict(hsi_dict, wavelengths, positions_dict):
    """Save HSI data to a .npz file."""
    dataname = os.path.basename(os.getcwd())
    target_dir = os.path.join("IMG", "STEADY_STATE")
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.join(target_dir, f"{dataname}_HSI_all_voltages.npz")

    mask = (wavelengths >= 400) & (wavelengths <= 1700)
    filtered_wavelengths = wavelengths[mask]

    filtered_hsi_dict = {
        voltage: hsi_dict[voltage][mask, :, :]
        for voltage in hsi_dict.keys()
    }

    np.savez_compressed(
        filename,
        **filtered_hsi_dict,
        wavelengths=filtered_wavelengths,
        positions=positions_dict
    )
    print(f"Saved filtered hsi_dict to: {filename}")

def load_hsi_dict():
    """Load HSI data from .npz file."""
    dataname = os.path.basename(os.getcwd())
    filename = os.path.join("IMG", "STEADY_STATE", f"{dataname}_HSI_all_voltages.npz")

    data = np.load(filename, allow_pickle=True)

    hsi_dict = {
        key: data[key]
        for key in data.files
        if key not in ['wavelengths', 'positions']
    }

    wavelengths = data['wavelengths']
    positions_dict = data['positions'].item() if 'positions' in data else {}

    print(f"Loaded hsi_dict from: {filename}")
    return hsi_dict, wavelengths, positions_dict

def load_single_image(file_path, dtype=np.int16):
    """Load a single image from a .bin file."""
    try:
        img = np.fromfile(file_path, dtype=dtype)
        V_size, H_size = int(img[0]), int(img[2])
        return np.reshape(img[4:], (V_size, H_size))
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None