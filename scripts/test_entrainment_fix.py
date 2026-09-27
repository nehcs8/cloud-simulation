#!/usr/bin/env python3
"""
Quick test to verify entrainment fix works.

Runs 3 short simulations (30 min each):
1. No entrainment
2. Weak entrainment (0.001 s^-1, ~1000s timescale)
3. Strong entrainment (0.01 s^-1, ~100s timescale)
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'no_entr',
        'enable': False,
        'rate': 0.0,
        'desc': 'No entrainment (baseline)'
    },
    {
        'name': 'weak_entr',
        'enable': True,
        'rate': 0.001,
        'desc': 'Weak entrainment (0.001 s^-1, ~1000s timescale)'
    },
    {
        'name': 'strong_entr',
        'enable': True,
        'rate': 0.01,
        'desc': 'Strong entrainment (0.01 s^-1, ~100s timescale)'
    },
]

SIM_DURATION = 1800.0  # 30 minutes
OUTPUT_EVERY = 60.0
DX = 100.0

def modify_config(enable, rate):
    """Modify config.py for test."""
    config_path = Path(__file__).parent.parent / "config.py"

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ENABLE_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_ENTRAINMENT: bool = {enable}\n")
        elif line.startswith('ENTRAINMENT_RATE:'):
            modified_lines.append(f"ENTRAINMENT_RATE: float = {rate}  # s^-1\n")
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

    print(f"✓ Config modified: ENABLE_ENTRAINMENT={enable}, RATE={rate:.4f} s^-1")

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

    print(f"✓ Moved outputs")

def extract_metrics(output_dir):
    """Extract final metrics."""
    diag_csv = output_dir / "diag.csv"
    if not diag_csv.exists():
        return None

    with open(diag_csv, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        header = lines[0].strip().split(',')
        data = lines[-1].strip().split(',')

    metrics = dict(zip(header, data))

    try:
        return {
            'qc_max': float(metrics.get('qc_max', 0)),
            'qc_mean': float(metrics.get('qc_mean', 0)),
            'cloud_frac': float(metrics.get('cloud_frac', 0)) * 100,
        }
    except (ValueError, KeyError):
        return None

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("ENTRAINMENT FIX TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/60:.0f} minutes per simulation")
    print(f"Total tests: {len(TESTS)}")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_entrainment_fix"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup_entr_fix"
    shutil.copy(config_path, config_backup)
    print(f"✓ Backed up config\n")

    results = []

    try:
        for test in TESTS:
            test_name = f"{date_prefix}_{test['name']}"
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# {test['desc']}")
            print(f"{'#'*70}\n")

            modify_config(test['enable'], test['rate'])
            success = run_simulation(test_name, output_dir)

            if success:
                move_outputs(output_dir)
                metrics = extract_metrics(output_dir)
                if metrics:
                    results.append({
                        'test': test['name'],
                        'desc': test['desc'],
                        **metrics
                    })

    finally:
        # Restore config
        shutil.copy(config_backup, config_path)
        config_backup.unlink()
        print(f"\n✓ Restored original config")

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for r in results:
        print(f"\n{r['desc']}:")
        print(f"  QC max: {r['qc_max']:.2f} g/kg")
        print(f"  QC mean: {r['qc_mean']:.4f} g/kg")
        print(f"  Cloud fraction: {r['cloud_frac']:.1f}%")
    print("="*70)

    # Check if fix worked
    if len(results) == 3:
        no_entr = results[0]
        weak_entr = results[1]
        strong_entr = results[2]

        print("\nDIFFERENCE FROM BASELINE:")
        print(f"  Weak entrainment:   QC max change = {weak_entr['qc_max'] - no_entr['qc_max']:+.2f} g/kg")
        print(f"  Strong entrainment: QC max change = {strong_entr['qc_max'] - no_entr['qc_max']:+.2f} g/kg")

        if abs(weak_entr['qc_max'] - no_entr['qc_max']) > 0.1:
            print("\n✓ FIX WORKS! Entrainment now has measurable effect.")
        else:
            print("\n✗ Still no effect. More investigation needed.")

    print(f"\nResults saved to: {base_dir}")

if __name__ == '__main__':
    main()
