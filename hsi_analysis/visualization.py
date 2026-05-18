import os
import numpy as np
from ipywidgets import Dropdown, IntSlider, Output, VBox, FloatSlider, HBox, Layout, Button
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from .core import create_asymmetric_window, compute_fft, calibrate_spectrum  # Import compute_fft and calibrate_spectrum
from .io import load_all_images, load_positions_dyn, load_calibration_dyn,load_positions_ss,load_calibration_ss

def compute_hsi_for_frame(frame_dir, positions, calibration_data):
    """Compute HSI for a single frame directory."""
    images = load_all_images(frame_dir)
    if images is None:
        return None

    wavelengths, pseudo_freqs = calibration_data
    V_size, H_size = images.shape[1], images.shape[2]
    hsi = np.empty((len(wavelengths), V_size, H_size), dtype=np.float32)

    for i in range(V_size):
        for j in range(H_size):
            amplitude, freq_axis, _ = compute_fft(images[:, i, j], positions)
            _, hsi[:, i, j] = calibrate_spectrum(amplitude, freq_axis, wavelengths, pseudo_freqs)
    return hsi

def compute_hsi_for_voltage_ss(voltage_dir, positions, calibration_data):
    """Compute HSI for a single voltage directory."""
    # Load images from STEPS folder
    images_dir = os.path.join(voltage_dir, "STEPS")
    images = load_all_images(images_dir)
    if images is None:
        return None

    wavelengths, pseudo_freqs = calibration_data
    V_size, H_size = images.shape[1], images.shape[2]
    hsi = np.empty((len(wavelengths), V_size, H_size), dtype=np.float32)

    for i in range(V_size):
        for j in range(H_size):
            amplitude, freq_axis, _ = compute_fft(images[:, i, j], positions)
            _, hsi[:, i, j] = calibrate_spectrum(amplitude, freq_axis, wavelengths, pseudo_freqs)
    return hsi

def compute_all_hsi_ss(data_dir):  # ✅ Now accepts data_dir
    """Main function to compute HSI for all voltages."""
    # Load calibration data
    calibration_data = load_calibration_ss()
    if calibration_data[0] is None:
        return {}, [], {}

    # Find all Vgs folders
    vgs_folders = [f for f in os.listdir(data_dir) if f.startswith('Vgs=')]
    if not vgs_folders:
        print(f"No Vgs folders found in {data_dir}")
        print(f"Contents: {os.listdir(data_dir) if os.path.exists(data_dir) else 'Directory not found'}")
        return {}, [], {}

    positions_dict = {}
    hsi_dict = {}

    for folder in vgs_folders:
        voltage_dir = os.path.join(data_dir, folder)
        voltage_key = folder.split('=')[1].replace('V', '')

        print(f"\nProcessing {folder}...")

        # Load positions for this voltage
        positions = load_positions_ss(voltage_dir)
        if positions is None:
            continue

        positions_dict[voltage_key] = positions

        # Compute HSI for this voltage
        try:
            hsi = compute_hsi_for_voltage_ss(voltage_dir, positions, calibration_data)
            if hsi is None:
                continue

            hsi_dict[voltage_key] = hsi
            print(f"Successfully processed {folder}")
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue

    if not hsi_dict:
        print("No valid data was processed")
        return {}, [], {}

    wavelengths = calibration_data[0]
    return hsi_dict, wavelengths, positions_dict  
    
def compute_all_hsi_dyn(data_dir, ref_frames_to_average=20):
    """Compute HSI for dynamic measurements (REF, ON, OFF frames)."""
    import glob
    import numpy as np

    # Load calibration from MAIN FOLDER
    calibration_data = load_calibration_dyn(data_dir)
    if calibration_data[0] is None:
        calibration_data = load_calibration_dyn(os.path.join(data_dir, 'IMG'))
        if calibration_data[0] is None:
            return {}, [], None

    positions = load_positions_dyn(data_dir)
    if positions is None:
        return {}, [], None

    hsi_dict = {}
    ref_success = 0
    on_success = 0
    off_success = 0

    # Process REF frames
    ref_dir = os.path.join(data_dir, 'REF')
    if not os.path.exists(ref_dir):
        ref_dir = os.path.join(data_dir, 'IMG', 'REF')

    if os.path.exists(ref_dir):
        ref_frames = sorted(glob.glob(os.path.join(ref_dir, 'FRAME_*')))
        if ref_frames:
            ref_hsi_list = []
            for frame_dir in ref_frames[:ref_frames_to_average]:
                frame_key = os.path.basename(frame_dir)
                try:
                    print(f"\rProcessing REF frame {frame_key}...", end="", flush=True)
                    hsi = compute_hsi_for_frame(frame_dir, positions, calibration_data)
                    if hsi is not None:
                        ref_hsi_list.append(hsi)
                        ref_success += 1
                except Exception as e:
                    print(f"\nError processing REF frame {frame_key}: {str(e)}")
                    break  # Stop at first error

            if ref_hsi_list:
                hsi_dict['REF_FRAME_AVE'] = np.mean(np.stack(ref_hsi_list), axis=0)

    # Process ON frames
    on_frames = sorted(glob.glob(os.path.join(data_dir, 'IMG', 'ON', 'FRAME_*')))
    error_occurred_on = False
    for frame_dir in on_frames:
        if error_occurred_on:
            break
        frame_key = os.path.basename(frame_dir)
        try:
            print(f"\rProcessing ON frame {frame_key}...", end="", flush=True)
            hsi = compute_hsi_for_frame(frame_dir, positions, calibration_data)
            if hsi is not None:
                hsi_dict[f"ON_{frame_key}"] = hsi
                on_success += 1
        except Exception as e:
            print(f"\nError processing ON frame {frame_key}: {str(e)}")
            error_occurred_on = True
            continue

    # Process OFF frames
    off_frames = sorted(glob.glob(os.path.join(data_dir, 'IMG', 'OFF', 'FRAME_*')))
    error_occurred_off = False
    for frame_dir in off_frames:
        if error_occurred_off:
            break
        frame_key = os.path.basename(frame_dir)
        try:
            print(f"\rProcessing OFF frame {frame_key}...", end="", flush=True)
            hsi = compute_hsi_for_frame(frame_dir, positions, calibration_data)
            if hsi is not None:
                hsi_dict[f"OFF_{frame_key}"] = hsi
                off_success += 1
        except Exception as e:
            print(f"\nError processing OFF frame {frame_key}: {str(e)}")
            error_occurred_off = True
            continue

    # Print summary
    if ref_success > 0:
        print(f"\nSuccessfully loaded {ref_success} REF frames.")
    if on_success > 0:
        print(f"Successfully loaded {on_success} ON frames.")
    if off_success > 0:
        print(f"Successfully loaded {off_success} OFF frames.")

    if not hsi_dict:
        print("No valid data was processed")
        return {}, [], None

    return hsi_dict, calibration_data[0], positions

    
def create_unified_widget(hsi_dict, wavelengths, positions_dict, base_dir):
    """Create an interactive widget for visualizing HSI data."""
    if not hsi_dict:
        print("No HSI data available")
        return

    voltages = sorted(hsi_dict.keys(), key=float)
    highest_voltage = voltages[-1]
    lowest_voltage = voltages[0]

    first_voltage = highest_voltage
    V_size, H_size = hsi_dict[first_voltage].shape[1], hsi_dict[first_voltage].shape[2]
    first_positions = list(positions_dict.values())[0]

    print("\n=== Dataset Information ===")
    print(f"Voltages: {len(voltages)} ({', '.join(voltages)})")
    print(f"Image size: {V_size}x{H_size} pixels")
    print(f"Wavelengths: {np.min(wavelengths):.1f} to {np.max(wavelengths):.1f} nm")

    default_x = H_size // 2
    default_y = V_size // 2

    voltage_dropdown = Dropdown(
        options=voltages,
        description='Voltage:',
        value=highest_voltage,
        layout=Layout(width='200px')
    )

    ref_voltage_dropdown = Dropdown(
        options=['None'] + voltages,
        description='Reference:',
        value=lowest_voltage,
        layout=Layout(width='200px')
    )

    x_slider = IntSlider(
        min=0, max=H_size - 1, value=default_x,
        description='X:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    y_slider = IntSlider(
        min=0, max=V_size - 1, value=default_y,
        description='Y:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    wavelength_slider = FloatSlider(
        min=np.min(wavelengths), max=np.max(wavelengths), value=800,
        step=10, description='Wavelength (nm):',
        layout=Layout(width='250px'),
        continuous_update=False
    )

    x_min_slider = FloatSlider(
        min=np.min(wavelengths), max=np.max(wavelengths), value=500,
        step=10, description='X min:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    x_max_slider = FloatSlider(
        min=np.min(wavelengths), max=np.max(wavelengths), value=1700,
        step=10, description='X max:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    img_min_slider = FloatSlider(
        min=-1, max=1, value=0,
        step=0.01, description='Img min:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    img_max_slider = FloatSlider(
        min=-1, max=1, value=1,
        step=0.01, description='Img max:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    spec_min_slider = FloatSlider(
        min=-1, max=1, value=0,
        step=0.01, description='Spec min:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    spec_max_slider = FloatSlider(
        min=-1, max=1, value=1,
        step=0.01, description='Spec max:',
        layout=Layout(width='200px'),
        continuous_update=False
    )

    autoscale_button = Button(
        description='Autoscale Both',
        layout=Layout(width='200px')
    )

    output = Output()

    def autoscale(b):
        voltage = voltage_dropdown.value
        ref_voltage = ref_voltage_dropdown.value
        wavelength_idx = np.argmin(np.abs(wavelengths - wavelength_slider.value))
        hsi = hsi_dict[voltage]

        img_data = hsi[wavelength_idx, :, :]
        if ref_voltage != 'None':
            ref_img = hsi_dict[ref_voltage][wavelength_idx, :, :]
            ref_img[ref_img <= 0] = 1e-6
            img_data[img_data <= 0] = 1e-6
            img_data = -np.log10(img_data / ref_img)
        img_min_slider.value = np.min(img_data)
        img_max_slider.value = np.max(img_data)

        x, y = x_slider.value, y_slider.value
        spectrum = hsi[:, y, x]
        if ref_voltage != 'None':
            ref_spectrum = hsi_dict[ref_voltage][:, y, x]
            ref_spectrum[ref_spectrum <= 0] = 1e-6
            spectrum[spectrum <= 0] = 1e-6
            spectrum = -np.log10(spectrum / ref_spectrum)
        spec_min_slider.value = np.min(spectrum)
        spec_max_slider.value = np.max(spectrum)

        update_displays(None)

    def update_displays(change):
        output.clear_output()

        voltage = voltage_dropdown.value
        ref_voltage = ref_voltage_dropdown.value
        x, y = x_slider.value, y_slider.value
        wavelength_idx = np.argmin(np.abs(wavelengths - wavelength_slider.value))
        hsi = hsi_dict[voltage]

        positions = positions_dict[voltage]
        voltage_dir = os.path.join(base_dir, f'Vgs={voltage}V')
        images = load_all_images(voltage_dir)
        if images is None:
            print(f"Failed to load images for {voltage}V")
            return

        raw_interferogram = images[:, y, x]
        processed_interferogram = raw_interferogram - np.mean(raw_interferogram)
        window = create_asymmetric_window(positions)
        processed_interferogram = processed_interferogram * window
        spectrum = hsi[:, y, x]

        if ref_voltage != 'None':
            ref_spectrum = hsi_dict[ref_voltage][:, y, x]
            ref_spectrum[ref_spectrum <= 0] = 1e-6
            spectrum[spectrum <= 0] = 1e-6
            spectrum = -np.log10(spectrum / ref_spectrum)
            display_mode = 'Absorption'
        else:
            display_mode = 'Intensity'

        with output:
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.15, 1.3], height_ratios=[1, 0.15], wspace=0.3, hspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            img_data = hsi[wavelength_idx, :, :]
            if ref_voltage != 'None':
                ref_img = hsi_dict[ref_voltage][wavelength_idx, :, :]
                ref_img[ref_img <= 0] = 1e-6
                img_data[img_data <= 0] = 1e-6
                img_data = -np.log10(img_data / ref_img)
                img = ax1.imshow(img_data, cmap='viridis', vmin=img_min_slider.value, vmax=img_max_slider.value)
            else:
                img = ax1.imshow(img_data, cmap='gray', vmin=img_min_slider.value, vmax=img_max_slider.value)
            ax1.scatter(x, y, color='white', s=200, edgecolor='red', linewidth=2)
            fig.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f'Image at {wavelengths[wavelength_idx]:.1f} nm (Vgs={voltage}V)')

            ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
            vertical_cut = img_data[:, x]
            ax2.plot(vertical_cut, range(len(vertical_cut)), 'r-', linewidth=1.5)
            ax2.set_xlim(img_min_slider.value, img_max_slider.value)
            ax2.set_title('Vertical')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            plt.setp(ax2.get_yticklabels(), visible=False)

            ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
            horizontal_cut = img_data[y, :]
            ax3.plot(range(len(horizontal_cut)), horizontal_cut, 'b-', linewidth=1.5)
            ax3.set_ylim(img_min_slider.value, img_max_slider.value)
            ax3.set_title('Horizontal')
            ax3.grid(True, alpha=0.3)

            gs_right = gs[:, 2].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
            ax4 = fig.add_subplot(gs_right[0])
            ax4.plot(wavelengths, spectrum, 'b-', linewidth=1)
            ax4.axvline(wavelengths[wavelength_idx], color='red', linestyle='--', alpha=0.7)
            ax4.set_ylim(spec_min_slider.value, spec_max_slider.value)
            ax4.set_xlim(x_min_slider.value, x_max_slider.value)
            ax4.set_title(f'{display_mode} at ({y}, {x})')
            ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs_right[1])
            ax5.plot(positions, raw_interferogram, 'gray', alpha=0.5, linewidth=1, label='Raw')
            ax5.plot(positions, processed_interferogram, 'b-', linewidth=1, label='Processed')
            window_scale = max(abs(raw_interferogram)) * 0.3
            ax5.plot(positions, window * window_scale, 'r--', alpha=0.5, linewidth=1, label='Window')
            zpd_index = np.argmin(np.abs(positions))
            ax5.axvline(positions[zpd_index], color='g', linestyle=':', alpha=0.7, label='ZPD')
            ax5.legend(loc='upper right', fontsize=8)
            ax5.grid(True, alpha=0.3)

            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.show()

    x_min_slider.observe(lambda change: on_x_min_change(change), names='value')
    x_max_slider.observe(lambda change: on_x_max_change(change), names='value')
    autoscale_button.on_click(autoscale)

    def on_x_min_change(change):
        if change['new'] >= x_max_slider.value:
            x_max_slider.value = change['new'] + 10
        update_displays(change)

    def on_x_max_change(change):
        if change['new'] <= x_min_slider.value:
            x_min_slider.value = change['new'] - 10
        update_displays(change)

    line1 = HBox([voltage_dropdown, ref_voltage_dropdown])
    line2 = HBox([x_slider, y_slider, wavelength_slider])
    line3 = HBox([x_min_slider, x_max_slider, img_min_slider, img_max_slider])
    line4 = HBox([spec_min_slider, spec_max_slider, autoscale_button])

    controls = VBox([line1, line2, line3, line4, output])

    display(controls)

    all_controls = [
        voltage_dropdown, ref_voltage_dropdown,
        x_slider, y_slider, wavelength_slider,
        x_min_slider, x_max_slider,
        img_min_slider, img_max_slider,
        spec_min_slider, spec_max_slider
    ]
    for control in all_controls:
        control.observe(update_displays, names='value')

    def update_slider_limits(change):
        voltage = voltage_dropdown.value
        V_size, H_size = hsi_dict[voltage].shape[1], hsi_dict[voltage].shape[2]
        x_slider.max = H_size - 1
        y_slider.max = V_size - 1
        if x_slider.value >= H_size:
            x_slider.value = H_size // 2
        if y_slider.value >= V_size:
            y_slider.value = V_size // 2

    voltage_dropdown.observe(update_slider_limits, names='value')

    autoscale(None)
    update_displays(None)

def create_time_resolved_widget_from_data(filtered_hsi_dict, wavelengths, positions):
    """
    Time-resolved widget with:
    - Frame selection
    - ON/OFF state selection
    - Interactive plots for images, cuts, and spectra
    """
    # Extract frame numbers and states
    frames = []
    states = []
    for key in filtered_hsi_dict.keys():
        match = re.match(r'(ON|OFF)_FRAME_(\d+)', key)
        if match:
            state, frame = match.groups()
            frames.append(frame)
            if state not in states:
                states.append(state)

    frames = sorted(list(set(frames)), key=lambda x: int(x))
    states = sorted(states)

    # Set default values
    default_frame = frames[len(frames)//2]
    default_state = 'ON' if 'ON' in states else states[0]
    first_key = f"{default_state}_FRAME_{default_frame}"
    V_size, H_size = filtered_hsi_dict[first_key].shape[1], filtered_hsi_dict[first_key].shape[2]
    default_x = H_size // 2
    default_y = V_size // 2

    # Widgets
    state_dropdown = widgets.Dropdown(
        options=states,
        description='State:',
        value=default_state,
        layout=widgets.Layout(width='150px')
    )

    frame_dropdown = widgets.Dropdown(
        options=frames,
        description='Frame:',
        value=default_frame,
        layout=widgets.Layout(width='150px')
    )

    x_slider = widgets.IntSlider(
        min=0, max=H_size - 1, value=default_x,
        description='X:',
        layout=widgets.Layout(width='210px'),
        continuous_update=False
    )

    y_slider = widgets.IntSlider(
        min=0, max=V_size - 1, value=default_y,
        description='Y:',
        layout=widgets.Layout(width='210px'),
        continuous_update=False
    )

    wavelength_slider = widgets.FloatSlider(
        min=np.min(wavelengths), max=np.max(wavelengths), value=np.mean(wavelengths),
        step=5, description='λ(nm):',
        layout=widgets.Layout(width='252px'),
        continuous_update=False
    )

    img_min_slider = widgets.FloatSlider(
        min=-1, max=1, value=0,
        step=0.01, description='Img min:',
        layout=widgets.Layout(width='168px'),
        continuous_update=False
    )

    img_max_slider = widgets.FloatSlider(
        min=-1, max=1, value=1,
        step=0.01, description='Img max:',
        layout=widgets.Layout(width='168px'),
        continuous_update=False
    )

    spec_min_slider = widgets.FloatSlider(
        min=-1, max=1, value=0,
        step=0.01, description='Spec min:',
        layout=widgets.Layout(width='168px'),
        continuous_update=False
    )

    spec_max_slider = widgets.FloatSlider(
        min=-1, max=1, value=1,
        step=0.01, description='Spec max:',
        layout=widgets.Layout(width='168px'),
        continuous_update=False
    )

    output = widgets.Output()

    # --- Update Function ---
    def update_displays(state=state_dropdown.value, frame=frame_dropdown.value,
                       x=x_slider.value, y=y_slider.value, wavelength=wavelength_slider.value,
                       img_min=img_min_slider.value, img_max=img_max_slider.value,
                       spec_min=spec_min_slider.value, spec_max=spec_max_slider.value):
        output.clear_output()
        current_key = f"{state}_FRAME_{frame}"
        hsi = filtered_hsi_dict[current_key]
        wavelength_idx = np.argmin(np.abs(wavelengths - wavelength))

        spectrum = hsi[:, y, x]

        with output:
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.15, 1.3],
                                 height_ratios=[1, 0.15], wspace=0.3, hspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            img_data = hsi[wavelength_idx, :, :]
            img = ax1.imshow(img_data, cmap='viridis',
                           vmin=img_min, vmax=img_max)
            ax1.scatter(x, y, color='white', s=200, edgecolor='red', linewidth=2)
            fig.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f'Image at {wavelengths[wavelength_idx]:.1f} nm\n{current_key}')

            ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
            vertical_cut = img_data[:, x]
            ax2.plot(vertical_cut, range(len(vertical_cut)), 'r-', linewidth=1.5)
            ax2.set_xlim(img_min, img_max)
            ax2.set_title('Vertical')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            plt.setp(ax2.get_yticklabels(), visible=False)

            ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
            horizontal_cut = img_data[y, :]
            ax3.plot(range(len(horizontal_cut)), horizontal_cut, 'b-', linewidth=1.5)
            ax3.set_ylim(img_min, img_max)
            ax3.set_title('Horizontal')
            ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(gs[:, 2])
            ax4.plot(wavelengths, spectrum, 'b-', linewidth=1)
            ax4.axvline(wavelength, color='red', linestyle='--', alpha=0.7)
            ax4.set_ylim(spec_min, spec_max)
            ax4.set_xlim(500, 1350)
            ax4.set_title(f'Spectrum at ({y}, {x})')
            ax4.grid(True, alpha=0.3)

            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.show()

    # --- Link Widgets to Update Function ---
    widgets.interact(
        update_displays,
        state=state_dropdown,
        frame=frame_dropdown,
        x=x_slider,
        y=y_slider,
        wavelength=wavelength_slider,
        img_min=img_min_slider,
        img_max=img_max_slider,
        spec_min=spec_min_slider,
        spec_max=spec_max_slider
    )

    # --- Display the Widget ---
    display(widgets.VBox([
        widgets.HBox([state_dropdown, frame_dropdown]),
        widgets.HBox([x_slider, y_slider]),
        widgets.HBox([wavelength_slider]),
        widgets.HBox([img_min_slider, img_max_slider, spec_min_slider, spec_max_slider]),
        output
    ]))

    # --- Initial Display ---
    update_displays()

#  Widget Update and Display Functions
def get_current_key(state_dropdown, frame_dropdown):
    """Get current key from state and frame dropdowns."""
    return f"{state_dropdown.value}_FRAME_{frame_dropdown.value}"

def autoscale(state_dropdown, frame_dropdown, wavelength_slider, x_slider, y_slider,
              img_min_slider, img_max_slider, spec_min_slider, spec_max_slider, hsi_dict, wavelengths, update_displays):
    """Autoscale both image and spectrum."""
    current_key = get_current_key(state_dropdown, frame_dropdown)
    hsi = hsi_dict[current_key]

    wavelength_idx = np.argmin(np.abs(wavelengths - wavelength_slider.value))

    # Autoscale image
    img_data = hsi[wavelength_idx, :, :]
    img_min_slider.value = np.min(img_data)
    img_max_slider.value = np.max(img_data)

    # Autoscale spectrum
    x, y = x_slider.value, y_slider.value
    spectrum = hsi[:, y, x]
    spec_min_slider.value = np.min(spectrum)
    spec_max_slider.value = np.max(spectrum)

    update_displays(None)

def update_displays(state_dropdown, frame_dropdown, x_slider, y_slider, wavelength_slider,
                    x_min_slider, x_max_slider, img_min_slider, img_max_slider,
                    spec_min_slider, spec_max_slider, hsi_dict, wavelengths, positions, output):
    """Update all displays."""
    output.clear_output()

    current_key = get_current_key(state_dropdown, frame_dropdown)
    hsi = hsi_dict[current_key]
    x, y = x_slider.value, y_slider.value
    wavelength_idx = np.argmin(np.abs(wavelengths - wavelength_slider.value))

    # Get spectrum
    spectrum = hsi[:, y, x]

    with output:
        # Create figure with proper layout
        fig = plt.figure(figsize=(16, 9))

        # Main gridspec
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.15, 1.3],
                             height_ratios=[1, 0.15], wspace=0.3, hspace=0.3)

        # Image plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        img_data = hsi[wavelength_idx, :, :]
        img = ax1.imshow(img_data, cmap='viridis',
                       vmin=img_min_slider.value, vmax=img_max_slider.value)

        # Large white pixel marker
        ax1.scatter(x, y, color='white', s=200, edgecolor='red', linewidth=2)
        fig.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f'Image at {wavelengths[wavelength_idx]:.1f} nm\n{current_key}')

        # Vertical cut (top middle)
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        vertical_cut = img_data[:, x]
        ax2.plot(vertical_cut, range(len(vertical_cut)), 'r-', linewidth=1.5)
        ax2.set_xlim(img_min_slider.value, img_max_slider.value)
        ax2.set_title('Vertical')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        plt.setp(ax2.get_yticklabels(), visible=False)

        # Horizontal cut (bottom left)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        horizontal_cut = img_data[y, :]
        ax3.plot(range(len(horizontal_cut)), horizontal_cut, 'b-', linewidth=1.5)
        ax3.set_ylim(img_min_slider.value, img_max_slider.value)
        ax3.set_title('Horizontal')
        ax3.grid(True, alpha=0.3)

        # Spectrum plot (right column)
        ax4 = fig.add_subplot(gs[:, 2])
        ax4.plot(wavelengths, spectrum, 'b-', linewidth=1)
        ax4.axvline(wavelengths[wavelength_idx], color='red', linestyle='--', alpha=0.7)
        ax4.set_ylim(spec_min_slider.value, spec_max_slider.value)
        ax4.set_xlim(x_min_slider.value, x_max_slider.value)
        ax4.set_title(f'Spectrum at ({y}, {x})')
        ax4.grid(True, alpha=0.3)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()