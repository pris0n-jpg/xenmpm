#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Calibration Data Visualization Tool
Visualizes force_xyz[2] (Z-force) from real_calibration_data.pkl
Shows multiple runs of the same trajectory on the same plot with fitted curves
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from typing import Dict, List, Tuple, Optional
import sys

try:
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, polynomial fitting will be used instead")


class RealCalibrationDataVisualizer:
    def __init__(self, data_path: str):
        """
        Initialize the real calibration data visualizer

        Parameters:
        - data_path: Path to real_calibration_data.pkl
        """
        self.data_path = data_path
        self.data = self.load_data(data_path)

        # Get list of objects
        self.objects = sorted(list(self.data.keys()))

        if not self.objects:
            raise ValueError("No objects found in the data file")

        self.current_object = self.objects[0]
        self.current_trajectory = None
        self.trajectories = {}
        self.edit_mode = False
        self.selected_runs = {}  # Dictionary to store selected runs for deletion

        # Build trajectory dictionary for each object
        for obj in self.objects:
            # Extract base trajectory names (without run numbers)
            traj_names = set()
            for traj_key in self.data[obj].keys():
                # Extract base name by removing "_runX" suffix
                if "_run" in traj_key:
                    base_name = traj_key.split("_run")[0]
                else:
                    base_name = traj_key
                traj_names.add(base_name)

            self.trajectories[obj] = sorted(list(traj_names))

        self.current_trajectory = self.trajectories[self.current_object][0] if self.trajectories[self.current_object] else None

        # Setup the plot
        self.setup_plot()

    def load_data(self, file_path: str) -> Dict:
        """Load pickle data file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def extract_force_z(self, obj: str, base_traj: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract force_xyz[2] values and step numbers for all runs of a given base trajectory

        Returns:
        - Dict mapping run_name to (steps, forces) tuple
        - steps: array of step indices (starting from 1, not 0)
        - forces: array of Z-force values
        """
        runs_data = {}

        # Find all runs for this base trajectory
        for traj_key in self.data[obj].keys():
            if traj_key.startswith(base_traj + "_run") or traj_key == base_traj:
                traj_data = self.data[obj][traj_key]
                steps = []
                forces = []

                for step_key in sorted(traj_data.keys()):
                    step_data = traj_data[step_key]
                    if 'force_xyz' in step_data:
                        force_z = step_data['force_xyz'][2]
                        # Extract step number from key like "step_000", "step_001", etc.
                        step_num = int(step_key.split('_')[-1])

                        steps.append(step_num + 1)  # Start from 1 (step_000 -> 1)
                        forces.append(float(force_z))

                if len(steps) > 0:
                    runs_data[traj_key] = (np.array(steps), np.array(forces))

        return runs_data

    def fit_curve(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit a smooth curve through the data points and extrapolate to x=0

        Returns:
        - x_dense: dense x values for smooth plotting (including 0)
        - y_fitted: fitted y values
        - y_intercept: y value at x=0
        """
        if len(x) < 2:
            return x, y, y[0] if len(y) > 0 else 0

        # Create dense x range including 0 for extrapolation
        x_min, x_max = 0, x.max()
        x_dense = np.linspace(x_min, x_max, 200)

        if SCIPY_AVAILABLE and len(x) >= 4:
            try:
                # Use spline fitting
                k = min(3, len(x) - 1)  # Spline degree
                s = len(x) * 0.1  # Smoothing factor
                spline = UnivariateSpline(x, y, k=k, s=s)
                y_fitted = spline(x_dense)
                y_intercept = spline(0)
                return x_dense, y_fitted, y_intercept
            except:
                pass

        # Fallback to polynomial fitting
        try:
            degree = min(3, len(x) - 1)
            poly = np.polyfit(x, y, degree)
            y_fitted = np.polyval(poly, x_dense)
            y_intercept = np.polyval(poly, 0)
            return x_dense, y_fitted, y_intercept
        except:
            # If all else fails, just use linear interpolation
            return x, y, y[0] if len(y) > 0 else 0

    def setup_plot(self):
        """Setup the matplotlib figure and widgets"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Real Calibration Data Visualization')

        # Main plot area (larger)
        self.ax_main = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)

        # Object selection area (left column, top)
        self.ax_object = plt.subplot2grid((4, 4), (0, 0), rowspan=1)
        self.ax_object.set_title('Select Object', fontsize=10, fontweight='bold')

        # Trajectory selection area (left column, middle)
        self.ax_trajectory = plt.subplot2grid((4, 4), (1, 0), rowspan=1)
        self.ax_trajectory.set_title('Select Trajectory', fontsize=10, fontweight='bold')

        # Run selection area (left column, bottom)
        self.ax_runs = plt.subplot2grid((4, 4), (2, 0), rowspan=1)
        self.ax_runs.set_title('Select Runs to Delete', fontsize=10, fontweight='bold')

        # Button area (bottom)
        self.ax_buttons = plt.subplot2grid((4, 4), (3, 0), colspan=4)
        self.ax_buttons.axis('off')

        # Create radio buttons for object selection
        self.radio_object = RadioButtons(self.ax_object, self.objects)
        self.radio_object.on_clicked(self.on_object_change)

        # Create radio buttons for trajectory selection
        self.update_trajectory_buttons()

        # Create checkboxes for run selection
        self.update_run_checkboxes()

        # Create buttons
        self.create_buttons()

        # Plot initial data
        self.update_plot()

        plt.tight_layout()

    def update_trajectory_buttons(self):
        """Update trajectory radio buttons based on current object"""
        self.ax_trajectory.clear()
        self.ax_trajectory.set_title('Select Trajectory', fontsize=10, fontweight='bold')

        trajs = self.trajectories[self.current_object]
        if trajs:
            self.radio_trajectory = RadioButtons(self.ax_trajectory, trajs)
            self.radio_trajectory.on_clicked(self.on_trajectory_change)
            if self.current_trajectory not in trajs:
                self.current_trajectory = trajs[0]
        else:
            self.ax_trajectory.text(0.5, 0.5, 'No trajectories',
                                   ha='center', va='center', transform=self.ax_trajectory.transAxes)
            self.current_trajectory = None

    def update_run_checkboxes(self):
        """Update run checkboxes based on current trajectory"""
        self.ax_runs.clear()
        self.ax_runs.set_title('Select Runs to Delete', fontsize=10, fontweight='bold')

        if self.current_trajectory:
            # Get all runs for this trajectory
            runs = []
            for traj_key in self.data[self.current_object].keys():
                if traj_key.startswith(self.current_trajectory + "_run"):
                    runs.append(traj_key)

            if runs:
                # Initialize selected runs if not exists
                if self.current_trajectory not in self.selected_runs:
                    self.selected_runs[self.current_trajectory] = [False] * len(runs)

                self.check_runs = CheckButtons(self.ax_runs, runs, self.selected_runs[self.current_trajectory])
                self.check_runs.on_clicked(self.on_run_select)
            else:
                self.ax_runs.text(0.5, 0.5, 'No runs found',
                                 ha='center', va='center', transform=self.ax_runs.transAxes)
        else:
            self.ax_runs.text(0.5, 0.5, 'Select a trajectory',
                             ha='center', va='center', transform=self.ax_runs.transAxes)

    def create_buttons(self):
        """Create control buttons"""
        # Edit mode button
        ax_edit = plt.axes([0.1, 0.02, 0.1, 0.03])
        self.btn_edit = Button(ax_edit, 'Toggle Edit')
        self.btn_edit.on_clicked(self.toggle_edit_mode)

        # Delete selected runs button
        ax_delete = plt.axes([0.25, 0.02, 0.1, 0.03])
        self.btn_delete = Button(ax_delete, 'Delete Selected')
        self.btn_delete.on_clicked(self.delete_selected_runs)

        # Save data button
        ax_save = plt.axes([0.4, 0.02, 0.1, 0.03])
        self.btn_save = Button(ax_save, 'Save Data')
        self.btn_save.on_clicked(self.save_data)

        # Refresh button
        ax_refresh = plt.axes([0.55, 0.02, 0.1, 0.03])
        self.btn_refresh = Button(ax_refresh, 'Refresh')
        self.btn_refresh.on_clicked(self.refresh_data)

    def on_object_change(self, label):
        """Callback when object selection changes"""
        self.current_object = label
        self.update_trajectory_buttons()
        self.update_plot()
        plt.draw()

    def on_trajectory_change(self, label):
        """Callback when trajectory selection changes"""
        self.current_trajectory = label
        self.update_run_checkboxes()
        self.update_plot()
        plt.draw()

    def on_run_select(self, label):
        """Callback when run checkbox is clicked"""
        if self.current_trajectory not in self.selected_runs:
            return

        # Get all runs for this trajectory
        runs = []
        for traj_key in self.data[self.current_object].keys():
            if traj_key.startswith(self.current_trajectory + "_run"):
                runs.append(traj_key)

        # Update selection state
        if label in runs:
            idx = runs.index(label)
            self.selected_runs[self.current_trajectory][idx] = not self.selected_runs[self.current_trajectory][idx]

    def toggle_edit_mode(self, event):
        """Toggle edit mode"""
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.btn_edit.label.set_text('Exit Edit')
            self.ax_runs.set_visible(True)
        else:
            self.btn_edit.label.set_text('Toggle Edit')
            # Clear all selections
            for traj in self.selected_runs:
                self.selected_runs[traj] = [False] * len(self.selected_runs[traj])
            self.update_run_checkboxes()

        plt.draw()

    def delete_selected_runs(self, event):
        """Delete selected runs from data and renumber remaining runs"""
        if not self.edit_mode:
            return

        if self.current_trajectory not in self.selected_runs:
            return

        # Get all runs for this trajectory, sorted by run number
        runs = []
        for traj_key in self.data[self.current_object].keys():
            if traj_key.startswith(self.current_trajectory + "_run"):
                runs.append(traj_key)

        # Sort runs by their run number for proper renumbering
        def get_run_number(run_name):
            try:
                return int(run_name.split("_run")[-1])
            except:
                return -1
        runs.sort(key=get_run_number)

        # Find which runs to delete and their positions
        runs_to_delete = []
        for i, run in enumerate(runs):
            if i < len(self.selected_runs[self.current_trajectory]) and self.selected_runs[self.current_trajectory][i]:
                runs_to_delete.append(run)

        if not runs_to_delete:
            print("No runs selected for deletion")
            return

        # Get the smallest run number being deleted
        deleted_run_numbers = [get_run_number(run) for run in runs_to_delete]
        min_deleted_run = min(deleted_run_numbers)

        # Delete selected runs
        deleted_count = 0
        for run in runs_to_delete:
            if run in self.data[self.current_object]:
                del self.data[self.current_object][run]
                deleted_count += 1

        # Renumber runs that come after the smallest deleted run number
        if min_deleted_run >= 0:  # Valid run number
            # Find all runs that need renumbering
            runs_to_renumber = []
            for traj_key in list(self.data[self.current_object].keys()):
                if traj_key.startswith(self.current_trajectory + "_run"):
                    run_num = get_run_number(traj_key)
                    if run_num > min_deleted_run:
                        runs_to_renumber.append((run_num, traj_key))

            # Sort by run number (ascending) to renumber properly
            runs_to_renumber.sort()

            # Renumber runs
            renumbered_count = 0
            for old_run_num, old_run_name in runs_to_renumber:
                # Calculate new run number
                # Count how many runs with smaller numbers were deleted
                smaller_deleted = sum(1 for num in deleted_run_numbers if num < old_run_num)
                new_run_num = old_run_num - smaller_deleted

                # Create new run name
                new_run_name = f"{self.current_trajectory}_run{new_run_num}"

                # Move data to new key
                if old_run_name in self.data[self.current_object]:
                    self.data[self.current_object][new_run_name] = self.data[self.current_object][old_run_name]
                    del self.data[self.current_object][old_run_name]
                    renumbered_count += 1

            if renumbered_count > 0:
                print(f"Renumbered {renumbered_count} runs after deletion")

        print(f"Deleted {deleted_count} runs from {self.current_trajectory}")

        # Clear selections and update UI
        self.selected_runs[self.current_trajectory] = [False] * len(runs)
        self.update_run_checkboxes()
        self.update_plot()
        plt.draw()

    def save_data(self, event):
        """Save modified data back to file"""
        try:
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Data saved to {self.data_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def refresh_data(self, event):
        """Reload data from file"""
        try:
            self.data = self.load_data(self.data_path)
            self.selected_runs = {}
            self.update_run_checkboxes()
            self.update_plot()
            plt.draw()
            print("Data refreshed from file")
        except Exception as e:
            print(f"Error refreshing data: {e}")

    def update_plot(self):
        """Update the main plot with current selection"""
        self.ax_main.clear()

        if self.current_trajectory is None:
            self.ax_main.text(0.5, 0.5, 'No trajectory selected',
                            ha='center', va='center', transform=self.ax_main.transAxes,
                            fontsize=14)
            return

        try:
            # Extract force data for all runs of this trajectory
            runs_data = self.extract_force_z(self.current_object, self.current_trajectory)

            if not runs_data:
                self.ax_main.text(0.5, 0.5, 'No force data available for this trajectory',
                                ha='center', va='center', transform=self.ax_main.transAxes,
                                fontsize=14)
                return

            # Color map for different runs
            colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

            # Get all runs for this trajectory
            runs = []
            for traj_key in self.data[self.current_object].keys():
                if traj_key.startswith(self.current_trajectory + "_run"):
                    runs.append(traj_key)

            # Plot each run
            for i, (run_name, (steps, forces)) in enumerate(runs_data.items()):
                color = colors[i % len(colors)]

                # Check if this run is selected for deletion
                is_selected = False
                if self.current_trajectory in self.selected_runs and run_name in runs:
                    idx = runs.index(run_name)
                    if idx < len(self.selected_runs[self.current_trajectory]):
                        is_selected = self.selected_runs[self.current_trajectory][idx]

                # Use different style if selected for deletion
                alpha = 0.3 if is_selected else 0.7
                linestyle = '--' if is_selected else '-'
                marker = 'x' if is_selected else 'o'

                # Plot data points
                self.ax_main.scatter(steps, forces,
                                   c=[color], s=60, alpha=alpha,
                                   edgecolors='black', linewidth=1,
                                   marker=marker,
                                   label=f'{run_name} Data', zorder=3)

                # Fit and plot curve
                if len(steps) >= 2:
                    x_fit, y_fit, y_intercept = self.fit_curve(steps, forces)
                    self.ax_main.plot(x_fit, y_fit, linestyle,
                                    color=color, linewidth=2, alpha=alpha,
                                    label=f'{run_name} Fitted')

                    # Mark y-intercept (x=0)
                    self.ax_main.plot(0, y_intercept, marker,
                                    color=color, markersize=8,
                                    markeredgecolor='black', markeredgewidth=1.5,
                                    zorder=4)

                    # Add annotation for y-intercept
                    self.ax_main.annotate(f'{y_intercept:.3f}',
                                         xy=(0, y_intercept),
                                         xytext=(5, 5), textcoords='offset points',
                                         fontsize=8, color=color,
                                         bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='white', alpha=0.7))

            # Formatting
            self.ax_main.set_xlabel('Step Number (starting from 1)', fontsize=12, fontweight='bold')
            self.ax_main.set_ylabel('Z-Force (N)', fontsize=12, fontweight='bold')

            title = f'Force Data: {self.current_object} - {self.current_trajectory}'
            if self.edit_mode:
                title += ' [EDIT MODE]'
            self.ax_main.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Add legend with smaller font
            self.ax_main.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
            self.ax_main.grid(True, alpha=0.3, linestyle='--')

            # Set X-axis to start from 0 to show y-intercept
            x_min, x_max = self.ax_main.get_xlim()
            self.ax_main.set_xlim(0, max(x_max, 5))  # Ensure at least some range

            # Add vertical line at x=0 to highlight y-intercept
            self.ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            # Add text annotation for y-intercept line
            self.ax_main.text(0.02, 0.98, 'Y-axis intercept (x=0)',
                             transform=self.ax_main.transAxes,
                             fontsize=9, verticalalignment='top',
                             bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='lightgray', alpha=0.5))

            # Add edit mode instructions
            if self.edit_mode:
                self.ax_main.text(0.98, 0.02,
                                 'Edit Mode: Select runs in left panel and click "Delete Selected"',
                                 transform=self.ax_main.transAxes,
                                 fontsize=9, horizontalalignment='right',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='yellow', alpha=0.5))

        except Exception as e:
            self.ax_main.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
                            ha='center', va='center', transform=self.ax_main.transAxes,
                            fontsize=12, color='red')

    def show(self):
        """Display the interactive plot"""
        plt.show()


def main():
    """Main function"""
    print("🎯 Real Calibration Data Visualization Tool")
    print("=" * 60)

    # Get script directory
    script_dir = Path(__file__).parent

    # Default path
    data_path = script_dir / "real_calibration_data.pkl"

    # Check if file exists
    if not data_path.exists():
        print(f"❌ Error: real_calibration_data.pkl not found at {data_path}")
        print(f"   Please ensure the file exists in: {script_dir}")
        sys.exit(1)

    print(f"✓ Loading data from: {data_path}")

    try:
        visualizer = RealCalibrationDataVisualizer(str(data_path))

        print(f"\n📊 Visualization ready!")
        print(f"   Objects available: {', '.join(visualizer.objects)}")
        total_trajs = sum(len(v) for v in visualizer.trajectories.values())
        print(f"   Total base trajectories: {total_trajs}")
        print(f"\n💡 Instructions:")
        print(f"   - Use left panel radio buttons to switch between objects")
        print(f"   - Use left middle panel to select base trajectories")
        print(f"   - Click 'Toggle Edit' to enable deletion mode")
        print(f"   - In edit mode, select runs to delete with checkboxes")
        print(f"   - Click 'Delete Selected' to remove selected runs")
        print(f"   - Click 'Save Data' to save changes to file")
        print(f"   - Click 'Refresh' to reload data from file")
        print(f"   - Each run is shown with a different color")
        print(f"   - Fitted curves extrapolate to x=0 (y-intercept marked)")
        print(f"\n🖥️  Opening visualization window...")

        visualizer.show()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()