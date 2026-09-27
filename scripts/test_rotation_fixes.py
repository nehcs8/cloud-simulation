#!/usr/bin/env python3
"""
Automated testing script for rotation fixes.

Runs 4 test configurations serially:
1. baseline (no fixes)
2. coriolis (Coriolis force only)
3. weno (WENO momentum only)
4. allfixes (Coriolis + WENO)

Each test:
- Modifies config flags
- Runs 2hr simulation (100m resolution)
- Generates GIF with vorticity overlay
- Creates diagnostic plots
- Saves results to experiments/rotation_tests/YYYYMMDD_rot_XXX/
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg

# Test matrix with date prefix
TEST_CONFIGS = [
    {
        'name': 'baseline',
        'enable_coriolis': False,
        'enable_weno': False,
        'description': 'Baseline (no fixes)'
    },
    {
        'name': 'coriolis',
        'enable_coriolis': True,
        'enable_weno': False,
        'description': 'Coriolis force only'
    },
    {
        'name': 'weno',
        'enable_coriolis': False,
        'enable_weno': True,
        'description': 'WENO momentum only'
    },
    {
        'name': 'allfixes',
        'enable_coriolis': True,
        'enable_weno': True,
        'description': 'All fixes (Coriolis + WENO)'
    },
]

# Simulation parameters
SIM_DURATION = 7200.0  # 2 hours in seconds
OUTPUT_EVERY = 60.0    # 1 minute
DX = 100.0             # 100m resolution
DOMAIN_SIZE = 12000.0  # 12 km

def modify_config(test_config):
    """Modify config.py with test-specific flags."""
    config_path = Path(__file__).parent.parent / "config.py"

    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # Modify relevant lines
    modified_lines = []
    for line in lines:
        if line.startswith('ENABLE_CORIOLIS:'):
            modified_lines.append(f"ENABLE_CORIOLIS: bool = {test_config['enable_coriolis']}\n")
        elif line.startswith('ENABLE_WENO_MOMENTUM:'):
            modified_lines.append(f"ENABLE_WENO_MOMENTUM: bool = {test_config['enable_weno']}\n")
        elif line.startswith('T_END:'):
            modified_lines.append(f"T_END: float = {SIM_DURATION}  # s - {SIM_DURATION/3600:.1f} hours\n")
        elif line.startswith('OUTPUT_EVERY:'):
            modified_lines.append(f"OUTPUT_EVERY: float = {OUTPUT_EVERY}  # s\n")
        elif line.startswith('DX:'):
            modified_lines.append(f"DX: float = {DX}  # m\n")
        elif line.startswith('DY:'):
            modified_lines.append(f"DY: float = {DX}  # m\n")
        elif line.startswith('RESTART_FROM:'):
            modified_lines.append(f"RESTART_FROM: str | None = None  # Fresh start\n")
        else:
            modified_lines.append(line)

    # Write modified config
    with open(config_path, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Modified config: Coriolis={test_config['enable_coriolis']}, WENO={test_config['enable_weno']}")

def run_simulation(test_name, output_dir):
    """Run simulation and move outputs to test directory."""
    print(f"\n{'='*70}")
    print(f"Running simulation: {test_name}")
    print(f"{'='*70}\n")

    # Run simulation
    main_py = Path(__file__).parent.parent / "main.py"
    log_file = output_dir / "simulation.log"

    try:
        with open(log_file, 'w') as log:
            result = subprocess.run(
                [sys.executable, str(main_py)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=main_py.parent,
                check=True
            )
        print(f"✓ Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed with error code {e.returncode}")
        print(f"  See log: {log_file}")
        return False

def move_outputs(output_dir):
    """Move snapshots and diag.csv to test directory."""
    base_dir = Path(__file__).parent.parent

    # Move snapshots
    snapshots_src = base_dir / "snapshots"
    snapshots_dst = output_dir / "snapshots"
    if snapshots_src.exists():
        shutil.move(str(snapshots_src), str(snapshots_dst))
        print(f"✓ Moved snapshots to {snapshots_dst}")

    # Move diag.csv
    diag_src = base_dir / "diag.csv"
    diag_dst = output_dir / "diag.csv"
    if diag_src.exists():
        shutil.move(str(diag_src), str(diag_dst))
        print(f"✓ Moved diag.csv to {diag_dst}")

def generate_gif(output_dir):
    """Generate animated GIF from snapshots."""
    print("\nGenerating GIF...")

    viz_script = Path(__file__).parent / "visualize" / "visualize3d.py"
    if not viz_script.exists():
        print(f"✗ Visualization script not found: {viz_script}")
        return False

    gif_path = output_dir / "clouds.gif"

    try:
        subprocess.run(
            [sys.executable, str(viz_script),
             "--snapshots", str(output_dir / "snapshots"),
             "--output", str(gif_path)],
            check=True,
            cwd=viz_script.parent
        )
        print(f"✓ Generated GIF: {gif_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ GIF generation failed")
        return False

def create_diagnostic_plots(output_dir):
    """Create time series and trajectory plots."""
    print("\nCreating diagnostic plots...")

    diag_csv = output_dir / "diag.csv"
    if not diag_csv.exists():
        print(f"✗ diag.csv not found")
        return

    # Load diagnostics using numpy
    data = np.genfromtxt(diag_csv, delimiter=',', names=True)

    # Convert to dict for easier access
    df = {name: data[name] for name in data.dtype.names}

    # 1. Vorticity time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df['t'] / 60, df['vort_max'], 'b-', linewidth=2, label='vort_max')
    ax1.plot(df['t'] / 60, df['vort_cloud_mean'], 'r-', linewidth=2, label='vort_cloud_mean')
    ax1.axhline(y=0.01, color='k', linestyle='--', alpha=0.5, label='threshold (0.01 s⁻¹)')
    ax1.set_ylabel('Vorticity (s⁻¹)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Vorticity Evolution', fontsize=14, fontweight='bold')

    ax2.plot(df['t'] / 60, df['qc_max'], 'g-', linewidth=2, label='qc_max')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Cloud water (g/kg)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Cloud Development', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "vorticity_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created vorticity time series plot")

    # 2. Centroid trajectory
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filter out zero centroids (no clouds)
    mask = (df['centroid_x'] > 0) & (df['centroid_y'] > 0)
    x = df['centroid_x'][mask]
    y = df['centroid_y'][mask]
    t = df['t'][mask] / 60  # minutes

    if len(x) > 1:
        # Color by time
        scatter = ax.scatter(x, y, c=t, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)
        ax.plot(x[0], y[0], 'go', markersize=15, label='Start', markeredgecolor='k', markeredgewidth=2)
        ax.plot(x[-1], y[-1], 'ro', markersize=15, label='End', markeredgecolor='k', markeredgewidth=2)

        cbar = plt.colorbar(scatter, ax=ax, label='Time (minutes)')
        ax.legend(loc='upper right', fontsize=12)

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title('Cloud Centroid Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / "centroid_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created centroid trajectory plot")

def create_comparison_grid(base_dir):
    """Create 4-panel comparison grid from all tests."""
    print("\nCreating comparison grid...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    date_prefix = datetime.now().strftime("%Y%m%d")

    for idx, test_config in enumerate(TEST_CONFIGS):
        test_name = f"{date_prefix}_rot_{test_config['name']}"
        output_dir = base_dir / test_name
        vort_plot = output_dir / "vorticity_timeseries.png"

        if vort_plot.exists():
            img = plt.imread(vort_plot)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(test_config['description'], fontsize=16, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'Plot not available',
                          ha='center', va='center', fontsize=14)
            axes[idx].axis('off')

    plt.tight_layout()
    comparison_path = base_dir / f"{date_prefix}_comparison_grid.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created comparison grid: {comparison_path}")

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("ROTATION FIX AUTOMATED TESTING")
    print("="*70)
    print(f"Test configurations: {len(TEST_CONFIGS)}")
    print(f"Simulation duration: {SIM_DURATION/3600:.1f} hours")
    print(f"Resolution: {DX}m")
    print(f"Domain size: {DOMAIN_SIZE/1000:.1f} km")
    print("="*70 + "\n")

    # Create base output directory with date
    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / "rotation_tests"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Backup original config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup"
    shutil.copy(config_path, config_backup)
    print(f"✓ Backed up config to {config_backup}\n")

    results = []

    try:
        for test_config in TEST_CONFIGS:
            test_name = f"{date_prefix}_rot_{test_config['name']}"
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# TEST: {test_config['description']}")
            print(f"# Output: {output_dir}")
            print(f"{'#'*70}\n")

            # Modify config
            modify_config(test_config)

            # Run simulation
            success = run_simulation(test_name, output_dir)

            if success:
                # Move outputs
                move_outputs(output_dir)

                # Generate GIF (optional - may fail if visualize script has issues)
                generate_gif(output_dir)

                # Create diagnostic plots
                create_diagnostic_plots(output_dir)

                results.append({'test': test_name, 'status': 'SUCCESS'})
            else:
                results.append({'test': test_name, 'status': 'FAILED'})

        # Create comparison grid
        create_comparison_grid(base_dir)

    finally:
        # Restore original config
        shutil.copy(config_backup, config_path)
        config_backup.unlink()
        print(f"\n✓ Restored original config")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status_icon} {result['test']}: {result['status']}")
    print("="*70)

    print(f"\nAll results saved to: {base_dir}")
    print("\nDone!")

if __name__ == '__main__':
    main()
