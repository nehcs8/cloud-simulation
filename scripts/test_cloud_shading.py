#!/usr/bin/env python3
"""
Test cloud-radiative feedback (cloud shading).

Compares 3 simulations (2 hours each):
1. No shading (baseline) - dynamic entrainment only
2. Moderate shading (extinction=150, min_trans=0.2) - typical cumulus
3. Strong shading (extinction=200, min_trans=0.1) - opaque clouds

All tests use weak dynamic entrainment (0.001 s^-1) to allow cloud development.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'no_shading',
        'enable_shading': False,
        'extinction': 150.0,
        'min_trans': 0.2,
        'desc': 'No shading (baseline, dynamic entrainment only)'
    },
    {
        'name': 'moderate_shading',
        'enable_shading': True,
        'extinction': 150.0,
        'min_trans': 0.2,
        'desc': 'Moderate shading (extinction=150, min_trans=0.2)'
    },
    {
        'name': 'strong_shading',
        'enable_shading': True,
        'extinction': 200.0,
        'min_trans': 0.1,
        'desc': 'Strong shading (extinction=200, min_trans=0.1)'
    },
]

SIM_DURATION = 7200.0  # 2 hours
OUTPUT_EVERY = 60.0
DX = 100.0
ENTRAINMENT_RATE = 0.001  # s^-1, weak

def modify_config(enable_shading, extinction, min_trans):
    """Modify config.py for test."""
    config_path = Path(__file__).parent.parent / "config.py"

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ENABLE_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_ENTRAINMENT: bool = True\n")
        elif line.startswith('ENTRAINMENT_RATE:'):
            modified_lines.append(f"ENTRAINMENT_RATE: float = {ENTRAINMENT_RATE}  # s^-1\n")
        elif line.startswith('ENABLE_DYNAMIC_ENTRAINMENT:'):
            modified_lines.append(f"ENABLE_DYNAMIC_ENTRAINMENT: bool = True\n")
        elif line.startswith('ENABLE_CLOUD_SHADING:'):
            modified_lines.append(f"ENABLE_CLOUD_SHADING: bool = {enable_shading}\n")
        elif line.startswith('CLOUD_EXTINCTION:'):
            modified_lines.append(f"CLOUD_EXTINCTION: float = {extinction}  # m²/kg\n")
        elif line.startswith('MIN_TRANSMITTANCE:'):
            modified_lines.append(f"MIN_TRANSMITTANCE: float = {min_trans}\n")
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

    print(f"✓ Config: shading={enable_shading}, extinction={extinction:.0f}, min_trans={min_trans:.2f}")

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
    """Extract final metrics and time series statistics."""
    diag_csv = output_dir / "diag.csv"
    if not diag_csv.exists():
        return None

    with open(diag_csv, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        header = lines[0].strip().split(',')

        # Get final state
        final_data = lines[-1].strip().split(',')
        final_metrics = dict(zip(header, final_data))

        # Get time series for oscillation analysis
        qc_max_series = []
        cloud_frac_series = []
        for line in lines[1:]:
            data = line.strip().split(',')
            row = dict(zip(header, data))
            try:
                qc_max_series.append(float(row.get('qc_max', 0)))
                cloud_frac_series.append(float(row.get('cloud_frac', 0)))
            except ValueError:
                pass

    import numpy as np

    try:
        # Time-averaged metrics
        qc_max_mean = np.mean(qc_max_series)
        qc_max_std = np.std(qc_max_series)
        cloud_frac_mean = np.mean(cloud_frac_series) * 100
        cloud_frac_std = np.std(cloud_frac_series) * 100

        return {
            'qc_max_final': float(final_metrics.get('qc_max', 0)),
            'qc_max_mean': qc_max_mean,
            'qc_max_std': qc_max_std,
            'cloud_frac_mean': cloud_frac_mean,
            'cloud_frac_std': cloud_frac_std,
            'depth': float(final_metrics.get('CT', 0)) - float(final_metrics.get('CB', 0)),
        }
    except (ValueError, KeyError):
        return None

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("CLOUD-RADIATIVE FEEDBACK TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/3600:.0f} hours per simulation")
    print(f"Total tests: {len(TESTS)}")
    print(f"Dynamic entrainment: ON (0.001 s^-1)")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_cloud_shading"
    base_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = Path(__file__).parent.parent / "cloud_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup_shading_test"
    shutil.copy(config_path, config_backup)
    print(f"✓ Backed up config\n")

    results = []

    try:
        for i, test in enumerate(TESTS, 1):
            test_name = test['name']
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# TEST {i}/{len(TESTS)}: {test['desc']}")
            print(f"{'#'*70}\n")

            modify_config(test['enable_shading'], test['extinction'], test['min_trans'])
            success = run_simulation(test_name, output_dir)

            if success:
                move_outputs(output_dir)

                # Generate GIF
                gif_path = gif_dir / f"{date_prefix}_shading_{test_name}.gif"
                generate_gif(output_dir / "snapshots", gif_path)

                metrics = extract_metrics(output_dir)
                if metrics:
                    results.append({
                        'name': test['name'],
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
    print("RESULTS COMPARISON")
    print("="*70)

    if results:
        print(f"\n{'Test':<20} {'QC mean±std':<20} {'CloudFrac±std':<20} {'Depth':<10}")
        print("-" * 70)

        for r in results:
            qc_str = f"{r['qc_max_mean']:.2f}±{r['qc_max_std']:.2f}"
            cf_str = f"{r['cloud_frac_mean']:.1f}±{r['cloud_frac_std']:.1f}%"
            print(f"{r['name']:<20} {qc_str:<20} {cf_str:<20} {r['depth']:<10.0f}")

        print("\n" + "="*70)
        print("OSCILLATION ANALYSIS:")
        print("="*70)

        baseline = results[0]  # no shading
        print(f"\nBaseline (no shading) std: {baseline['qc_max_std']:.2f} g/kg")

        for r in results[1:]:
            ratio = r['qc_max_std'] / baseline['qc_max_std'] if baseline['qc_max_std'] > 0 else 0
            print(f"{r['name']:<20} std: {r['qc_max_std']:.2f} g/kg ({ratio:.1f}× baseline)")

        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("="*70)
        print("\nHigher std = stronger oscillations (pulsating clouds)")
        print("Lower std = steady-state behavior")

        max_std = max(r['qc_max_std'] for r in results)
        best = max(results, key=lambda r: r['qc_max_std'])

        if best['name'] != 'no_shading':
            print(f"\n✓ Cloud shading increases oscillations!")
            print(f"  Strongest pulsation: {best['desc']}")
        else:
            print(f"\n→ Shading doesn't increase oscillations (may need parameter tuning)")

    print(f"\n{'='*70}")
    print(f"Results saved to: {base_dir}")
    print(f"GIFs saved to: {gif_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
