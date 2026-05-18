import h5py


# saving functions
def save_hsi_data_hdf5(filtered_hsi_dict, wavelengths, positions, filepath):
    """Save the filtered and downsampled data to an HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        for key, data in filtered_hsi_dict.items():
            f.create_dataset(key, data=data, compression="gzip")
        f.create_dataset('wavelengths', data=wavelengths)
        f.create_dataset('positions', data=positions)
    print(f"Filtered data saved to {filepath} (HDF5, compressed)")

def load_hsi_data_hdf5(filepath):
    """Load the filtered and downsampled data from an HDF5 file."""
    filtered_hsi_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key not in ['wavelengths', 'positions']:
                filtered_hsi_dict[key] = f[key][()]
        wavelengths = f['wavelengths'][()]
        positions = f['positions'][()]
    print(f"Filtered data loaded from {filepath} (HDF5)")
    return filtered_hsi_dict, wavelengths, positions