#!/usr/bin/env python3
"""
Test updraft-rain interaction: can strong updrafts suspend raindrops?

This tests the critical fix where sedimentation now uses V_net = V_t - w
instead of just V_t.

Expected behavior:
- Strong updraft cores (w > V_t) accumulate rain → bigger clouds
- Weak updrafts (w < V_t) let rain fall → precipitation
- This should create realistic fluffy cumulus!

Single 2-hour test with moderate autoconversion (QC0=1.0 g/kg).
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

SIM_DURATION = 7200.0  # 2 hours
OUTPUT_EVERY = 60.0
DX = 100.0

def modify_config():
    """Set up config for test."""
    config_path = Path(__file__).parent.parent / "config.py"

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ENABLE_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_ENTRAINMENT: bool = True\n")
        elif line.startswith('ENTRAINMENT_RATE:'):
            modified_lines.append(f"ENTRAINMENT_RATE: float = 0.001  # s^-1, weak\n")
        elif line.startswith('ENABLE_DYNAMIC_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_DYNAMIC_ENTRAINMENT: bool = True\n")
        elif line.startswith('ENABLE_CLOUD_SHADING:'):
            modified_lines.append(f"ENABLE_CLOUD_SHADING: bool = True\n")
        elif line.startswith('CLOUD_EXTINCTION:'):
            modified_lines.append(f"CLOUD_EXTINCTION: float = 200.0  # m²/kg, strong\n")
        elif line.startswith('MIN_TRANSMITTANCE:'):
            modified_lines.append(f"MIN_TRANSMITTANCE: float = 0.1\n")
        elif line.startswith('T_END:'):
            modified_lines.append(f"T_END: float = {SIM_DURATION}  # s\n")
        elif line.startswith('OUTPUT_EVERY:'):
            modified_lines.append(f"OUTPUT_EVERY: float = {OUTPUT_EVERY}  # s\n")
        elif line.startswith('DX:'):
            modified_lines.append(f"DX: float = {DX}  # m\n")
        elif line.startswith('DY:'):
            modified_lines.append(f"DY: float = {DX}  # m\n")
        elif line.startswith('RESTART_FROM:'):
            modified_lines.append(f"RESTART_FROM: str | None = None\n")
        else:
            modified_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Config: 2hr simulation with all features enabled")

def modify_microphysics():
    """Set moderate autoconversion threshold."""
    micro_path = Path(__file__).parent.parent / "microphysics.py"

    with open(micro_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('_K1    ='):
            modified_lines.append(f"_K1    = 1e-3    # s^-1,  autoconversion rate\n")
        elif line.startswith('_QC0   ='):
            modified_lines.append(f"_QC0   = 1e-3    # kg/kg, autoconversion threshold (1.0 g/kg)\n")
        else:
            modified_lines.append(line)

    with open(micro_path, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Microphysics: QC0 = 1.0 g/kg, updraft suspension ENABLED")

def run_simulation(output_dir):
    """Run simulation."""
    print(f"\n{'='*70}")
    print(f"Running: Updraft-rain suspension test")
    print(f"{'='*70}\n")

    main_py = Path(__file__).parent.parent / "main.py"
    log_file = output_dir / "simulation.log"

    try:
        with open(log_file, 'w') as log:
            subprocess.run(
                [sys.executable, str(main_py)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=main_py.parent,
                check=True
            )
        print(f"✓ Simulation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed: {e.returncode}")
        return False

def move_outputs(output_dir):
    """Move outputs to test directory."""
    base_dir = Path(__file__).parent.parent

    snapshots_src = base_dir / "snapshots"
    if snapshots_src.exists():
        shutil.move(str(snapshots_src), str(output_dir / "snapshots"))

    diag_src = base_dir / "diag.csv"
    if diag_src.exists():
        shutil.move(str(diag_src), str(output_dir / "diag.csv"))

    print(f"✓ Moved outputs")

def generate_gif(snapshots_dir, output_gif):
    """Generate GIF from snapshots."""
    visualize_script = Path(__file__).parent / "visualize" / "visualize3d.py"

    try:
        subprocess.run(
            [sys.executable, str(visualize_script),
             "--snapshot-dir", str(snapshots_dir),
             "--out", str(output_gif)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ Generated GIF: {output_gif.name}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ GIF generation failed")
        return False

def analyze_results(output_dir):
    """Analyze cloud growth metrics."""
    diag_csv = output_dir / "diag.csv"
    if not diag_csv.exists():
        return

    with open(diag_csv, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return
        header = lines[0].strip().split(',')

        # Extract time series
        qc_max_series = []
        qr_max_series = []
        w_max_series = []
        for line in lines[1:]:
            data = line.strip().split(',')
            row = dict(zip(header, data))
            try:
                qc_max_series.append(float(row.get('qc_max', 0)))
                qr_max_series.append(float(row.get('qr_max', 0)))
                w_max_series.append(float(row.get('w_max', 0)))
            except ValueError:
                pass

    import numpy as np

    print("\n" + "="*70)
    print("RESULTS: Updraft-Rain Suspension")
    print("="*70)
    print(f"\nCloud water (QC):")
    print(f"  Max:  {np.max(qc_max_series):.2f} g/kg")
    print(f"  Mean: {np.mean(qc_max_series):.2f} g/kg")
    print(f"\nRain water (QR):")
    print(f"  Max:  {np.max(qr_max_series):.2f} g/kg")
    print(f"  Mean: {np.mean(qr_max_series):.2f} g/kg")
    print(f"\nUpdrafts (W):")
    print(f"  Max:  {np.max(w_max_series):.2f} m/s")
    print(f"  Mean: {np.mean(w_max_series):.2f} m/s")
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("="*70)
    print("\nWith updraft suspension:")
    print("  → Strong updrafts (w > 5-9 m/s) should hold rain")
    print("  → Clouds accumulate more water (higher QC + QR)")
    print("  → Bigger, fluffier cumulus clouds!")
    print(f"\n{'='*70}\n")

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("UPDRAFT-RAIN SUSPENSION TEST")
    print("="*70)
    print(f"Duration: 2 hours")
    print(f"\nPhysics fix: Rain sedimentation now uses V_net = V_t - w")
    print(f"  → Strong updrafts can suspend raindrops")
    print(f"  → Clouds can grow bigger!\n")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_updraft_suspension"
    base_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = Path(__file__).parent.parent / "cloud_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    micro_path = Path(__file__).parent.parent / "microphysics.py"
    config_backup = config_path.parent / "config.py.backup_updraft"
    micro_backup = micro_path.parent / "microphysics.py.backup_updraft"
    shutil.copy(config_path, config_backup)
    shutil.copy(micro_path, micro_backup)
    print(f"✓ Backed up config and microphysics\n")

    try:
        modify_config()
        modify_microphysics()
        success = run_simulation(base_dir)

        if success:
            move_outputs(base_dir)

            # Generate GIF
            gif_path = gif_dir / f"{date_prefix}_updraft_suspension.gif"
            generate_gif(base_dir / "snapshots", gif_path)

            analyze_results(base_dir)

    finally:
        # Restore files
        shutil.copy(config_backup, config_path)
        shutil.copy(micro_backup, micro_path)
        config_backup.unlink()
        micro_backup.unlink()
        print(f"\n✓ Restored original files")

    print(f"\nResults saved to: {base_dir}")
    print(f"GIF saved to: {gif_dir}\n")

if __name__ == '__main__':
    main()
