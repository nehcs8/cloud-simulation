#!/usr/bin/env python3
"""
Extended entrainment sweep test to find optimal parameter value.

Tests 6 entrainment rates from 0.0 to 0.01 s^-1:
  - 0.0000 s^-1: No entrainment (baseline)
  - 0.0005 s^-1: Very weak (~2000s timescale)
  - 0.0010 s^-1: Weak (~1000s timescale)
  - 0.0020 s^-1: Moderate (~500s timescale)
  - 0.0050 s^-1: Strong (~200s timescale)
  - 0.0100 s^-1: Very strong (~100s timescale)

Each simulation: 2 hours, 100m resolution
Total runtime: ~12 hours (6 tests × 2 hours each)
Output: experiments/YYYYMMDD_entrainment_sweep/
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'none',
        'enable': False,
        'rate': 0.0,
        'desc': 'No entrainment (baseline)'
    },
    {
        'name': 'very_weak',
        'enable': True,
        'rate': 0.0005,
        'desc': 'Very weak (0.0005 s^-1, ~2000s timescale)'
    },
    {
        'name': 'weak',
        'enable': True,
        'rate': 0.001,
        'desc': 'Weak (0.001 s^-1, ~1000s timescale)'
    },
    {
        'name': 'moderate',
        'enable': True,
        'rate': 0.002,
        'desc': 'Moderate (0.002 s^-1, ~500s timescale)'
    },
    {
        'name': 'strong',
        'enable': True,
        'rate': 0.005,
        'desc': 'Strong (0.005 s^-1, ~200s timescale)'
    },
    {
        'name': 'very_strong',
        'enable': True,
        'rate': 0.01,
        'desc': 'Very strong (0.01 s^-1, ~100s timescale)'
    },
]

SIM_DURATION = 7200.0  # 2 hours
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
            'CB': float(metrics.get('CB', 0)),
            'CT': float(metrics.get('CT', 0)),
            'depth': float(metrics.get('CT', 0)) - float(metrics.get('CB', 0)),
        }
    except (ValueError, KeyError):
        return None

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("EXTENDED ENTRAINMENT SWEEP TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/3600:.0f} hours per simulation")
    print(f"Total tests: {len(TESTS)}")
    print(f"Estimated total time: {len(TESTS) * SIM_DURATION/3600:.0f} hours")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_entrainment_sweep"
    base_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = Path(__file__).parent.parent / "cloud_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup_entr_sweep"
    shutil.copy(config_path, config_backup)
    print(f"✓ Backed up config\n")

    results = []

    try:
        for i, test in enumerate(TESTS, 1):
            test_name = f"entr_{test['name']}"
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# TEST {i}/{len(TESTS)}: {test['desc']}")
            print(f"{'#'*70}\n")

            modify_config(test['enable'], test['rate'])
            success = run_simulation(test_name, output_dir)

            if success:
                move_outputs(output_dir)

                # Generate GIF
                gif_path = gif_dir / f"{date_prefix}_{test_name}.gif"
                generate_gif(output_dir / "snapshots", gif_path)

                metrics = extract_metrics(output_dir)
                if metrics:
                    results.append({
                        'test': test['name'],
                        'rate': test['rate'],
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
    print("RESULTS SUMMARY")
    print("="*70)

    if results:
        # Print table header
        print(f"\n{'Rate (s^-1)':<12} {'QC max':<10} {'QC mean':<10} {'Depth':<10} {'Coverage':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['rate']:<12.4f} {r['qc_max']:<10.2f} {r['qc_mean']:<10.4f} {r['depth']:<10.0f} {r['cloud_frac']:<10.1f}%")

        # Calculate differences from baseline
        if len(results) > 1:
            baseline = results[0]
            print("\n" + "="*70)
            print("EFFECT RELATIVE TO BASELINE:")
            print("="*70)
            for r in results[1:]:
                qc_change = r['qc_max'] - baseline['qc_max']
                qc_pct = (qc_change / baseline['qc_max'] * 100) if baseline['qc_max'] > 0 else 0
                depth_change = r['depth'] - baseline['depth']
                print(f"\n{r['desc']}:")
                print(f"  QC max change:  {qc_change:+.2f} g/kg ({qc_pct:+.0f}%)")
                print(f"  Depth change:   {depth_change:+.0f} m")
                print(f"  Coverage change: {r['cloud_frac'] - baseline['cloud_frac']:+.1f}%")

    print(f"\n{'='*70}")
    print(f"Results saved to: {base_dir}")
    print(f"GIFs saved to: {gif_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
