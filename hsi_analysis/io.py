import numpy as np
import os
import glob

def load_all_images(frame_dir, dtype=np.int16):
    """Load all images from a FRAME folder."""
    bin_files = sorted(glob.glob(os.path.join(frame_dir, '*.bin')))
    if not bin_files:
        print(f"No .bin files found in {frame_dir}")
        return None

    first_img = load_single_image(bin_files[0], dtype)
    if first_img is None:
        return None

    N, V_size, H_size = len(bin_files), *first_img.shape
    images = np.empty((N, V_size, H_size), dtype=dtype)

    for i, bin_file in enumerate(bin_files):
        img = load_single_image(bin_file, dtype)
        if img is None:
            print(f"Failed to load image {i}: {bin_file}")
            return None
        if img.shape != (V_size, H_size):
            print(f"Shape mismatch at image {i}: got {img.shape}, expected {(V_size, H_size)}")
            return None
        images[i] = img

    return images

def load_positions_ss(voltage_dir, dtype=np.float64):
    """Load positions from positions.txt in voltage directory."""
    positions_file = os.path.join(voltage_dir, 'positions.txt')
    try:
        return np.loadtxt(positions_file, dtype=dtype, delimiter='\t')
    except Exception as e:
        print(f"Error loading positions from {positions_file}: {str(e)}")
        return None

def load_positions_dyn(data_dir):
    """Load positions from dataname_POS.txt."""
    positions_file = glob.glob(os.path.join(data_dir, '*POS.txt'))[0]
    try:
        return np.loadtxt(positions_file, dtype=np.float64, delimiter='\t')
    except Exception as e:
        print(f"Error loading positions from {positions_file}: {str(e)}")
        return None

def load_calibration_ss(dtype=np.float64):
    """Load calibration from calibration.txt in STEADY_STATE folder."""
    script_dir = os.getcwd()
    calibration_file = os.path.join(script_dir, "IMG", "STEADY_STATE", 'calibration.txt')
    try:
        data = np.loadtxt(calibration_file, dtype=dtype, delimiter='\t')
        return data[0, :], data[1, :]
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