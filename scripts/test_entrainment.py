#!/usr/bin/env python3
"""
Test entrainment parameterization with side-by-side comparison.

Runs 2 simulations:
1. no_entrainment: ENABLE_ENTRAINMENT=False (baseline)
2. with_entrainment: ENABLE_ENTRAINMENT=True (new physics)

Each: 1 hour, 100m resolution, saves to experiments/entrainment_tests/
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'no_entrainment',
        'enable_entrainment': False,
        'description': 'Baseline (no entrainment)'
    },
    {
        'name': 'with_entrainment',
        'enable_entrainment': True,
        'description': 'With entrainment (wispy clouds)'
    },
]

SIM_DURATION = 3600.0  # 1 hour
OUTPUT_EVERY = 60.0
DX = 100.0

def modify_config(enable_entrainment):
    """Modify config.py for test."""
    config_path = Path(__file__).parent.parent / "config.py"

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ENABLE_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_ENTRAINMENT: bool = {enable_entrainment}\n")
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

    print(f"✓ Config modified: ENABLE_ENTRAINMENT={enable_entrainment}")

def run_simulation(test_name, output_dir):
    """Run simulation."""
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
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

    print(f"✓ Moved outputs to {output_dir}")

def generate_gif(output_dir):
    """Generate GIF."""
    print("\nGenerating GIF...")

    viz_script = Path(__file__).parent / "visualize" / "visualize3d.py"
    gif_path = output_dir / "clouds.gif"

    try:
        subprocess.run(
            [sys.executable, str(viz_script),
             "--snapshot-dir", str(output_dir / "snapshots"),
             "--out", str(gif_path)],
            check=True,
            cwd=viz_script.parent,
            capture_output=True
        )
        print(f"✓ Generated GIF: {gif_path.name}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ GIF generation failed")
        return False

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("ENTRAINMENT PARAMETERIZATION TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/60:.0f} minutes")
    print(f"Resolution: {DX}m")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / "entrainment_tests"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup_entrainment"
    shutil.copy(config_path, config_backup)
    print(f"✓ Backed up config\n")

    results = []

    try:
        for test in TESTS:
            test_name = f"{date_prefix}_entr_{test['name']}"
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# {test['description']}")
            print(f"# Output: {output_dir}")
            print(f"{'#'*70}\n")

            modify_config(test['enable_entrainment'])

            success = run_simulation(test_name, output_dir)

            if success:
                move_outputs(output_dir)
                generate_gif(output_dir)
                results.append({'test': test_name, 'status': 'SUCCESS'})
            else:
                results.append({'test': test_name, 'status': 'FAILED'})

    finally:
        # Restore config
        shutil.copy(config_backup, config_path)
        config_backup.unlink()
        print(f"\n✓ Restored original config")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for result in results:
        icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{icon} {result['test']}: {result['status']}")
    print("="*70)

    print(f"\nResults: {base_dir}")
    print("\nCompare the GIFs to see the effect of entrainment!")
    print("Expected: 'with_entrainment' should show wispy, evolving cloud edges.")

if __name__ == '__main__':
    main()
