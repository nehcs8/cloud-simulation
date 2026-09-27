#!/usr/bin/env python3
"""
Systematic parameter sweep to identify what controls cloud aspect ratio.

Tests 3 parameter sets (9 simulations total):

Set 1: Wind Shear (vary wind profile)
  - no_shear: U=2m/s uniform
  - moderate_shear: U=2→7m/s (+5m/s over 3km)
  - strong_shear: U=2→12m/s (+10m/s over 3km)

Set 2: Surface Forcing (vary SHF/LHF)
  - weak_forcing: SHF=50, LHF=25 W/m²
  - moderate_forcing: SHF=100, LHF=50 W/m²
  - strong_forcing: SHF=200, LHF=100 W/m²

Set 3: Entrainment (vary ENTRAINMENT_RATE)
  - no_entrainment: 0.0 m^-1
  - moderate_entrainment: 0.0002 m^-1
  - strong_entrainment: 0.0004 m^-1

Each simulation: 1 hour, 100m resolution
Output: experiments/YYYYMMDD_param_sweep/
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Baseline parameters (kept constant except for the varied parameter)
BASELINE = {
    'shf': 100.0,
    'lhf': 50.0,
    'u_shear': 5.0,  # m/s increase over 3 km
    'entrainment_rate': 0.0002,
    'enable_entrainment': True,
}

# Test matrix
TEST_SETS = {
    'wind_shear': [
        {'name': 'no_shear', 'u_shear': 0.0, 'desc': 'No wind shear (U=2m/s uniform)'},
        {'name': 'moderate_shear', 'u_shear': 5.0, 'desc': 'Moderate shear (+5m/s over 3km)'},
        {'name': 'strong_shear', 'u_shear': 10.0, 'desc': 'Strong shear (+10m/s over 3km)'},
    ],
    'surface_forcing': [
        {'name': 'weak_forcing', 'shf': 50.0, 'lhf': 25.0, 'desc': 'Weak forcing (50/25 W/m²)'},
        {'name': 'moderate_forcing', 'shf': 100.0, 'lhf': 50.0, 'desc': 'Moderate forcing (100/50 W/m²)'},
        {'name': 'strong_forcing', 'shf': 200.0, 'lhf': 100.0, 'desc': 'Strong forcing (200/100 W/m²)'},
    ],
    'entrainment': [
        {'name': 'no_entrainment', 'entrainment_rate': 0.0, 'enable_entrainment': False, 'desc': 'No entrainment'},
        {'name': 'moderate_entrainment', 'entrainment_rate': 0.0002, 'enable_entrainment': True, 'desc': 'Moderate entrainment (0.0002)'},
        {'name': 'strong_entrainment', 'entrainment_rate': 0.0004, 'enable_entrainment': True, 'desc': 'Strong entrainment (0.0004)'},
    ],
}

SIM_DURATION = 3600.0  # 1 hour
OUTPUT_EVERY = 60.0
DX = 100.0

def modify_config_and_state(params, base_dir):
    """Modify config.py and state.py for wind shear."""
    config_path = base_dir / "config.py"
    state_path = base_dir / "state.py"

    # Modify config.py
    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('T_END:'):
            modified_lines.append(f"T_END: float = {SIM_DURATION}  # s\n")
        elif line.startswith('OUTPUT_EVERY:'):
            modified_lines.append(f"OUTPUT_EVERY: float = {OUTPUT_EVERY}  # s\n")
        elif line.startswith('DX:'):
            modified_lines.append(f"DX: float = {DX}  # m\n")
        elif line.startswith('DY:'):
            modified_lines.append(f"DY: float = {DX}  # m\n")
        elif line.startswith('RESTART_FROM:'):
            modified_lines.append(f"RESTART_FROM: str | None = None\n")
        elif line.startswith('SHF:'):
            modified_lines.append(f"SHF: float = {params['shf']}  # W/m²\n")
        elif line.startswith('LHF:'):
            modified_lines.append(f"LHF: float = {params['lhf']}\n")
        elif line.startswith('ENABLE_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_ENTRAINMENT: bool = {params['enable_entrainment']}\n")
        elif line.startswith('ENTRAINMENT_RATE:'):
            modified_lines.append(f"ENTRAINMENT_RATE: float = {params['entrainment_rate']}\n")
        else:
            modified_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(modified_lines)

    # Modify state.py to add wind shear
    with open(state_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for i, line in enumerate(lines):
        if 'self.u[:] = cfg.U_GEO' in line:
            # Replace uniform wind with sheared profile
            if params['u_shear'] > 0:
                modified_lines.append(f"            # Wind shear: U increases with height\n")
                modified_lines.append(f"            z_profile = g.z  # height array\n")
                modified_lines.append(f"            u_profile = cfg.U_GEO + (z_profile / 3000.0) * {params['u_shear']}  # +{params['u_shear']}m/s at 3km\n")
                modified_lines.append(f"            self.u[:] = u_profile[np.newaxis, np.newaxis, :]\n")
            else:
                modified_lines.append(line)  # Keep uniform wind
        elif 'self.v[:] = cfg.V_GEO' in line:
            modified_lines.append(line)  # V stays uniform
        else:
            modified_lines.append(line)

    with open(state_path, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Config modified: SHF={params['shf']}, LHF={params['lhf']}, "
          f"U_shear={params['u_shear']}, Entrainment={params['entrainment_rate']}")

def run_simulation(test_name, output_dir, base_dir):
    """Run simulation."""
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
    print(f"{'='*70}\n")

    main_py = base_dir / "main.py"
    log_file = output_dir / "simulation.log"

    try:
        with open(log_file, 'w') as log:
            subprocess.run(
                [sys.executable, str(main_py)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=base_dir,
                check=True
            )
        print(f"✓ Simulation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed: {e.returncode}")
        return False

def move_outputs(output_dir, base_dir):
    """Move outputs to test directory."""
    snapshots_src = base_dir / "snapshots"
    if snapshots_src.exists():
        shutil.move(str(snapshots_src), str(output_dir / "snapshots"))

    diag_src = base_dir / "diag.csv"
    if diag_src.exists():
        shutil.move(str(diag_src), str(output_dir / "diag.csv"))

    print(f"✓ Moved outputs")

def generate_gif(output_dir, base_dir):
    """Generate GIF."""
    print("Generating GIF...")

    viz_script = base_dir / "scripts" / "visualize" / "visualize3d.py"
    gif_path = output_dir / "clouds.gif"

    try:
        subprocess.run(
            [sys.executable, str(viz_script),
             "--snapshot-dir", str(output_dir / "snapshots"),
             "--out", str(gif_path)],
            check=True,
            capture_output=True
        )
        print(f"✓ Generated GIF")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ GIF generation failed")
        return False

def extract_metrics(output_dir):
    """Extract key metrics from simulation."""
    diag_csv = output_dir / "diag.csv"
    if not diag_csv.exists():
        return None

    # Read last line
    with open(diag_csv, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        header = lines[0].strip().split(',')
        data = lines[-1].strip().split(',')

    metrics = dict(zip(header, data))

    # Extract key values
    try:
        return {
            'qc_max': float(metrics.get('qc_max', 0)),
            'w_max': float(metrics.get('w_max', 0)),
            'cloud_base': float(metrics.get('CB', 0)),
            'cloud_top': float(metrics.get('CT', 0)),
            'cloud_depth': float(metrics.get('CT', 0)) - float(metrics.get('CB', 0)),
            'cloud_frac': float(metrics.get('cloud_frac', 0)) * 100,
            'vort_cloud_mean': float(metrics.get('vort_cloud_mean', 0)),
        }
    except (ValueError, KeyError):
        return None

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("SYSTEMATIC PARAMETER SWEEP")
    print("="*70)
    print(f"Duration: {SIM_DURATION/60:.0f} minutes per simulation")
    print(f"Total simulations: 9 (3 sets × 3 variations)")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent
    sweep_dir = base_dir / "experiments" / f"{date_prefix}_param_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Backup config and state
    config_path = base_dir / "config.py"
    state_path = base_dir / "state.py"
    config_backup = base_dir / "config.py.backup_sweep"
    state_backup = base_dir / "state.py.backup_sweep"
    shutil.copy(config_path, config_backup)
    shutil.copy(state_path, state_backup)
    print(f"✓ Backed up config and state\n")

    all_results = []

    try:
        for set_name, tests in TEST_SETS.items():
            print(f"\n{'#'*70}")
            print(f"# SET: {set_name.upper().replace('_', ' ')}")
            print(f"{'#'*70}\n")

            for test in tests:
                # Build parameter dict (baseline + varied parameter)
                params = BASELINE.copy()
                params.update(test)

                test_name = f"{set_name}_{test['name']}"
                output_dir = sweep_dir / test_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n--- {test['desc']} ---")

                # Modify config and state
                modify_config_and_state(params, base_dir)

                # Run simulation
                success = run_simulation(test_name, output_dir, base_dir)

                if success:
                    move_outputs(output_dir, base_dir)
                    generate_gif(output_dir, base_dir)

                    # Extract metrics
                    metrics = extract_metrics(output_dir)
                    if metrics:
                        all_results.append({
                            'set': set_name,
                            'test': test['name'],
                            'desc': test['desc'],
                            **metrics
                        })
                        print(f"\nMetrics:")
                        print(f"  Cloud depth: {metrics['cloud_depth']:.0f} m")
                        print(f"  QC max: {metrics['qc_max']:.2f} g/kg")
                        print(f"  W max: {metrics['w_max']:.2f} m/s")
                        print(f"  Cloud fraction: {metrics['cloud_frac']:.1f}%")

    finally:
        # Restore originals
        shutil.copy(config_backup, config_path)
        shutil.copy(state_backup, state_path)
        config_backup.unlink()
        state_backup.unlink()
        print(f"\n✓ Restored original config and state")

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for result in all_results:
        print(f"\n{result['set'].upper()} - {result['test']}:")
        print(f"  Depth: {result['cloud_depth']:.0f}m  QC: {result['qc_max']:.2f}g/kg  "
              f"W: {result['w_max']:.2f}m/s  Cover: {result['cloud_frac']:.1f}%")
    print("="*70)

    print(f"\nAll results saved to: {sweep_dir}")
    print("\nCompare GIFs to see which parameter most affects cloud shape!")

if __name__ == '__main__':
    main()
