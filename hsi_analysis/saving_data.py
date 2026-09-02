import h5py


# saving functions
def save_hsi_data_hdf5(filtered_hsi_dict, wavelengths, positions_dict, filepath):
    """Save nested HSI data (Vds/Vgs) to HDF5."""
    with h5py.File(filepath, 'w') as f:
        # Save nested structure: Vds/Vgs/hsi_data
        for vds, vgs_dict in filtered_hsi_dict.items():
            vds_group = f.create_group(vds)  # Create Vds group
            for vgs, data in vgs_dict.items():
                vds_group.create_dataset(vgs, data=data, compression="gzip")  # Save Vgs data under Vds

        # Save nested positions: Vds/Vgs/positions
        positions_group = f.create_group("positions")
        for vds, vgs_dict in positions_dict.items():
            vds_pos_group = positions_group.create_group(vds)
            for vgs, pos_data in vgs_dict.items():
                vds_pos_group.create_dataset(vgs, data=pos_data)

        # Save wavelengths (unchanged)
        f.create_dataset('wavelengths', data=wavelengths)
    print(f"Filtered data saved to {filepath} (HDF5, compressed)")

def load_hsi_data_hdf5(filepath):
    """Load nested HSI data (Vds/Vgs) from HDF5."""
    filtered_hsi_dict = {}
    positions_dict = {}
    with h5py.File(filepath, 'r') as f:
        # Load nested HSI data
        for vds in f.keys():
            if vds not in ['wavelengths', 'positions']:  # Skip non-Vds groups
                filtered_hsi_dict[vds] = {}
                vds_group = f[vds]
                for vgs in vds_group.keys():
                    filtered_hsi_dict[vds][vgs] = vds_group[vgs][()]

        # Load nested positions
        if 'positions' in f:
            positions_group = f['positions']
            for vds in positions_group.keys():
                positions_dict[vds] = {}
                vds_pos_group = positions_group[vds]
                for vgs in vds_pos_group.keys():
                    positions_dict[vds][vgs] = vds_pos_group[vgs][()]

        # Load wavelengths (unchanged)
        wavelengths = f['wavelengths'][()]

    print(f"Filtered data loaded from {filepath} (HDF5)")
    return filtered_hsi_dict, wavelengths, positions_dict