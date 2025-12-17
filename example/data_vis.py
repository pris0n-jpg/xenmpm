# tasks.md: 阶段 1-5 的完整实现

import sys
import os
import pickle
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Agg' for headless
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

import cv2
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# Import MarkerInterpolator for target grid

# --- Configuration ---
# Always use file-based path detection to avoid environment-specific issues
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJ_DIR / 'xengym' / 'data' / 'obj'

# --- Global State ---
STATE = {
    "objects": [],
    "trajectories": {},
    "current_object_idx": 0,
    "current_trj_idx": 0,
    "current_frame_idx": 0,
    "total_frames": 0,
    "frame_data": {},
    "playing": False,
    "sensor_0_visible": True,
    "sensor_1_visible": True,
}

def load_available_data():
    """Scan data directory to find available objects and trajectories."""
    STATE["objects"] = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    if not STATE["objects"]:
        print(f"Error: No object data found in {DATA_ROOT}")
        return False

    for obj in STATE["objects"]:
        # Assume sensor_0 always exists to find trajectories.
        trj_path = DATA_ROOT / obj / 'sensor_0'
        if trj_path.exists():
            STATE["trajectories"][obj] = sorted([
                d.name for d in trj_path.iterdir() 
                if d.is_dir() and d.name.startswith('trj_')
            ])
        else:
            STATE["trajectories"][obj] = []
    
    print("Loaded available data.")
    print(f"Found {len(STATE['objects'])} objects with trajectories:")
    for obj in STATE["objects"]:
        print(f"  - {obj}: {len(STATE['trajectories'][obj])} trajectories")
    select_object(0)  # Select first object by default
    return True

def select_object(index_change):
    """Select object and update state."""
    if not STATE["objects"]:
        return
    new_idx = STATE["current_object_idx"] + index_change
    STATE["current_object_idx"] = new_idx % len(STATE["objects"])
    STATE["current_trj_idx"] = 0
    select_trajectory(0)
    current_obj = STATE["objects"][STATE["current_object_idx"]]
    print(f"Selected object: {current_obj}")

def select_trajectory(index_change):
    """Select trajectory and update state."""
    current_object = STATE["objects"][STATE["current_object_idx"]]
    available_trjs = STATE["trajectories"].get(current_object, [])
    if not available_trjs:
        STATE["total_frames"] = 0
        print(f"Warning: No trajectories found for object '{current_object}'")
        return

    new_idx = STATE["current_trj_idx"] + index_change
    STATE["current_trj_idx"] = new_idx % len(available_trjs)
    
    # Update total frames based on selected trajectory
    current_trj = available_trjs[STATE["current_trj_idx"]]
    frame_path = DATA_ROOT / current_object / 'sensor_0' / current_trj
    if frame_path.exists():
        STATE["total_frames"] = len(list(frame_path.glob('frame_*.pkl')))
    else:
        STATE["total_frames"] = 0
    
    STATE["current_frame_idx"] = 0
    load_current_frame_data()
    print(f"Selected trajectory: {current_trj} ({STATE['total_frames']} frames)")

def load_current_frame_data():
    """Load data for currently selected frame from .pkl file."""
    STATE["frame_data"] = {}
    if STATE["total_frames"] == 0:
        return

    current_object = STATE["objects"][STATE["current_object_idx"]]
    current_trj = STATE["trajectories"][current_object][STATE["current_trj_idx"]]
    
    for i in range(2):  # sensor_0 and sensor_1
        sensor_key = f'sensor_{i}'
        file_path = DATA_ROOT / current_object / sensor_key / current_trj / f'frame_{STATE["current_frame_idx"]}.pkl'
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    STATE["frame_data"][sensor_key] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                STATE["frame_data"][sensor_key] = None
        else:
            STATE["frame_data"][sensor_key] = None

def create_marker_plot(data, ax, marker_type):
    """Draw specific marker data as 2D scatter plot with actual frame dimensions."""
    ax.clear()
    # Set limits to actual frame dimensions: 17.2 x 28.5
    ax.set_xlim(-17.2/2, 17.2/2)
    ax.set_ylim(-28.5/2, 28.5/2)
    ax.invert_yaxis()  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    
    # Add frame boundary
    from matplotlib.patches import Rectangle
    frame = Rectangle((-17.2/2, -28.5/2), 17.2, 28.5, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(frame)
    
    # Create target grid for marker restoration
    marker_row_col = [20, 11]
    marker_dx_dy_mm = [1.31, 1.31]
    y_ed, x_ed = np.array(marker_dx_dy_mm) * (np.array(marker_row_col) - 1) / 2
    x = np.linspace(-x_ed, x_ed, marker_row_col[1])
    y = np.linspace(-y_ed, y_ed, marker_row_col[0])
    X, Y = np.meshgrid(x, y)
    target_grid = np.stack([X, Y], axis=2)
    
    # Unified marker size
    marker_size = 30
    
    # Display initial marker positions (blue, semi-transparent)
    init_x_coords = target_grid[:, :, 0].flatten()
    init_y_coords = target_grid[:, :, 1].flatten()
    ax.scatter(init_x_coords, init_y_coords, 
              c='blue', marker='o', s=marker_size, 
              alpha=0.3, edgecolors='navy', linewidth=0.5, label='Initial')
    
    # Display displacement arrows from initial positions
    scale = 10
    if marker_type in data and data[marker_type] is not None:
        marker_data = np.array(data[marker_type])
        if marker_data.size > 0:
            # Handle different marker data formats
            if len(marker_data.shape) == 3:
                # 3D array format: [:,:,0] is x, [:,:,1] is y
                if marker_type == 'marker_real':
                    # marker_real is displacement field
                    displacement_x = marker_data[:, :, 0].flatten()
                    displacement_y = marker_data[:, :, 1].flatten()
                    
                    # Remove invalid points
                    valid_mask = ~(np.isnan(displacement_x) | np.isnan(displacement_y))
                    valid_x_init = init_x_coords[valid_mask]
                    valid_y_init = init_y_coords[valid_mask]
                    valid_disp_x = displacement_x[valid_mask]
                    valid_disp_y = displacement_y[valid_mask]
                    
                    if len(valid_x_init) > 0:
                        # Draw displacement arrows with 3x magnification
                        ax.quiver(valid_x_init, valid_y_init, 
                                 valid_disp_x * scale, valid_disp_y * scale,
                                 angles='xy', scale_units='xy', scale=1,
                                 color='red', alpha=0.7, width=0.008,
                                 headwidth=2, headlength=2, headaxislength=1.5,
                                 label='Displacement (3x)') 
                
                else:  # marker_sim
                    # For marker_sim, extract coordinates directly
                    current_x = marker_data[:, :, 0].flatten()
                    current_y = marker_data[:, :, 1].flatten()
                    
                    # Remove invalid points
                    valid_mask = ~(np.isnan(current_x) | np.isnan(current_y))
                    if np.sum(valid_mask) > 0:
                        valid_current_x = current_x[valid_mask]
                        valid_current_y = current_y[valid_mask]
                        valid_x_init = init_x_coords[valid_mask]
                        valid_y_init = init_y_coords[valid_mask]
                        
                        # Center marker_sim at origin
                        center_x = np.mean(valid_current_x)
                        center_y = np.mean(valid_current_y)
                        centered_x = valid_current_x - center_x
                        centered_y = valid_current_y - center_y
                        
                        # Calculate displacement from initial to centered current positions
                        displacement_x = centered_x - valid_x_init
                        displacement_y = centered_y - valid_y_init
                        
                        # Filter displacements below threshold for sim marker
                        displacement_magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
                        threshold = 0.03  # 阈值，单位mm，可根据需要调整
                        above_threshold = displacement_magnitude > threshold
                        
                        if len(valid_x_init) > 0 and np.any(above_threshold):
                            # Only show arrows above threshold
                            filtered_x_init = valid_x_init[above_threshold]
                            filtered_y_init = valid_y_init[above_threshold]
                            filtered_disp_x = displacement_x[above_threshold]
                            filtered_disp_y = displacement_y[above_threshold]
                            
                            # Draw displacement arrows with scale magnification
                            ax.quiver(-filtered_x_init, -filtered_y_init,
                                     -filtered_disp_x * scale, -filtered_disp_y * scale,
                                     angles='xy', scale_units='xy', scale=1,
                                     color='red', alpha=0.7, width=0.008,
                                     headwidth=2, headlength=2, headaxislength=1.5,
                                     label=f'Displacement > {threshold}mm ({scale}x)')
    
    # Set title, labels and legend
    title_map = {'marker_real': 'Real Markers', 'marker_sim': 'Sim Markers'}
    ax.set_title(title_map.get(marker_type, 'Markers'))
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

def update_display():
    """Update display content."""
    if not STATE["frame_data"]:
        return
    
    # Clear all subplots
    for ax in axes.flatten():
        ax.clear()
    
    # Set title information with proper positioning
    current_object = STATE["objects"][STATE["current_object_idx"]]
    current_trj = STATE["trajectories"][current_object][STATE["current_trj_idx"]]
    fig.suptitle(f'Object: {current_object} | Trajectory: {current_trj} | Frame: {STATE["current_frame_idx"]+1}/{STATE["total_frames"]}', 
                 fontsize=14, y=0.98)
    
    # Data types and corresponding column positions (removed depth data)
    data_keys = ['real_rectify', 'diff_real', 'rectify_sim', 'diff_sim']
    
    for sensor_idx in range(2):  # sensor_0 and sensor_1
        sensor_key = f'sensor_{sensor_idx}'
        is_visible = STATE[f"sensor_{sensor_idx}_visible"]
        sensor_data = STATE["frame_data"].get(sensor_key)
        
        if not is_visible:
            continue
            
        if sensor_data is None:
            # If no data, show "No Data" in first column
            axes[sensor_idx, 0].text(0.5, 0.5, f'{sensor_key}\nNo Data', 
                                    ha='center', va='center', transform=axes[sensor_idx, 0].transAxes)
            continue
        
        # Display image data
        for col_idx, key in enumerate(data_keys):
            ax = axes[sensor_idx, col_idx]
            if key in sensor_data and sensor_data[key] is not None:
                img = sensor_data[key]
                
                # Handle color channels - real is BGR, sim is RGB
                if 'real' in key and len(img.shape) == 3:
                    # Real images are BGR, convert to RGB for display
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    # For grayscale images (diff), use grayscale colormap
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'{sensor_key}_{key}')
                    ax.axis('off')
                    continue
                
                # For RGB images (sim) or converted BGR->RGB (real)
                ax.imshow(img)
                ax.set_title(f'{sensor_key}_{key}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{sensor_key}_{key}')
            ax.axis('off')
        
        # Display marker data - Real markers in column 4, Sim markers in column 5
        create_marker_plot(sensor_data, axes[sensor_idx, 4], 'marker_real')
        create_marker_plot(sensor_data, axes[sensor_idx, 5], 'marker_sim')
    
    # Don't use tight_layout() as it conflicts with button controls
    fig.canvas.draw()

# Control functions
def next_frame(event=None):
    if STATE["total_frames"] > 0:
        STATE["current_frame_idx"] = (STATE["current_frame_idx"] + 1) % STATE["total_frames"]
        load_current_frame_data()
        update_display()
        print(f"Frame: {STATE['current_frame_idx']+1}/{STATE['total_frames']}")

def prev_frame(event=None):
    if STATE["total_frames"] > 0:
        STATE["current_frame_idx"] = (STATE["current_frame_idx"] - 1 + STATE["total_frames"]) % STATE["total_frames"]
        load_current_frame_data()
        update_display()
        print(f"Frame: {STATE['current_frame_idx']+1}/{STATE['total_frames']}")

def next_object(event=None):
    select_object(1)
    update_display()

def prev_object(event=None):
    select_object(-1)
    update_display()

def next_trajectory(event=None):
    select_trajectory(1)
    update_display()

def prev_trajectory(event=None):
    select_trajectory(-1)
    update_display()

def toggle_sensor_0(event=None):
    STATE["sensor_0_visible"] = not STATE["sensor_0_visible"]
    update_display()

def toggle_sensor_1(event=None):
    STATE["sensor_1_visible"] = not STATE["sensor_1_visible"]
    update_display()

def toggle_animation(event=None):
    global anim
    STATE["playing"] = not STATE["playing"]
    if STATE["playing"]:
        anim = FuncAnimation(fig, lambda frame: next_frame(), interval=200, repeat=True)
    else:
        if 'anim' in globals():
            anim.event_source.stop()

def on_key_press(event):
    """Handle keyboard events."""
    if event.key == 'right':
        next_frame()
    elif event.key == 'left':
        prev_frame()
    elif event.key == 'up':
        prev_object()
    elif event.key == 'down':
        next_object()
    elif event.key == 'j':
        prev_trajectory()
    elif event.key == 'k':
        next_trajectory()
    elif event.key == ' ':
        toggle_animation()
    elif event.key == '1':
        toggle_sensor_0()
    elif event.key == '2':
        toggle_sensor_1()
    elif event.key == 'q':
        plt.close('all')

def main():
    """Main function to set up and run the application."""
    global fig, axes
    
    if not DATA_ROOT.exists():
        print(f"Error: Data directory not found: {DATA_ROOT}")
        sys.exit(1)

    if not load_available_data():
        sys.exit(1)

    # Create matplotlib figure with space for controls
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 6, figure=fig, top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    # Create subplots
    axes = np.empty((2, 6), dtype=object)
    for i in range(2):
        for j in range(6):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    # Bind keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Create control buttons - adjusted for smaller figure width
    ax_prev_obj = plt.axes([0.02, 0.92, 0.09, 0.04])
    ax_next_obj = plt.axes([0.12, 0.92, 0.09, 0.04])
    ax_prev_trj = plt.axes([0.22, 0.92, 0.09, 0.04])
    ax_next_trj = plt.axes([0.32, 0.92, 0.09, 0.04])
    ax_prev_frame = plt.axes([0.42, 0.92, 0.09, 0.04])
    ax_next_frame = plt.axes([0.52, 0.92, 0.09, 0.04])
    ax_play = plt.axes([0.62, 0.92, 0.09, 0.04])
    ax_sensor0 = plt.axes([0.72, 0.92, 0.09, 0.04])
    ax_sensor1 = plt.axes([0.82, 0.92, 0.09, 0.04])

    Button(ax_prev_obj, 'Prev Obj').on_clicked(prev_object)
    Button(ax_next_obj, 'Next Obj').on_clicked(next_object)
    Button(ax_prev_trj, 'Prev Trj').on_clicked(prev_trajectory)
    Button(ax_next_trj, 'Next Trj').on_clicked(next_trajectory)
    Button(ax_prev_frame, '< Frame').on_clicked(prev_frame)
    Button(ax_next_frame, 'Frame >').on_clicked(next_frame)
    Button(ax_play, 'Play/Pause').on_clicked(toggle_animation)
    Button(ax_sensor0, 'Sensor0').on_clicked(toggle_sensor_0)
    Button(ax_sensor1, 'Sensor1').on_clicked(toggle_sensor_1)

    # Display control instructions
    print("\n--- XenGym Data Viewer Controls ---")
    print("Up/Down Arrows: Change Object")
    print("Left/Right Arrows: Change Frame")  
    print("J/K Keys: Change Trajectory")
    print("Spacebar: Play/Pause Animation")
    print("1: Toggle Sensor 0 Visibility")
    print("2: Toggle Sensor 1 Visibility")
    print("Q: Quit")
    print("Or use buttons on the interface")
    print("-----------------------------------\n")

    # Initial display
    update_display()
    
    plt.show()

if __name__ == "__main__":
    main()
