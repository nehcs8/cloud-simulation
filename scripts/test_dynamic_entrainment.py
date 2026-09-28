#!/usr/bin/env python3
"""
Test dynamic entrainment vs constant entrainment.

Compares 2 simulations (2 hours each, no cloud shading):
1. Constant entrainment (0.001 s^-1) - baseline
2. Dynamic entrainment (0.001 s^-1, scales with |w|) - velocity-dependent

Both tests use weak entrainment rate to highlight the dynamic effect.
Cloud shading is OFF to isolate the dynamic entrainment behavior.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'constant_entr',
        'enable_dynamic': False,
        'desc': 'Constant entrainment (0.001 s^-1)'
    },
    {
        'name': 'dynamic_entr',
        'enable_dynamic': True,
        'desc': 'Dynamic entrainment (0.001 s^-1, velocity-dependent)'
    },
]

SIM_DURATION = 7200.0  # 2 hours
OUTPUT_EVERY = 60.0
DX = 100.0
ENTRAINMENT_RATE = 0.001  # s^-1, weak entrainment

def modify_config(enable_dynamic):
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
            modified_lines.append(f"ENABLE_DYNAMIC_ENTRAINMENT: bool = {enable_dynamic}\n")
        elif line.startswith('ENABLE_CLOUD_SHADING:'):
            modified_lines.append(f"ENABLE_CLOUD_SHADING: bool = False  # OFF for this test\n")
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

    print(f"✓ Config modified: ENABLE_DYNAMIC_ENTRAINMENT={enable_dynamic}")

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
    print("DYNAMIC ENTRAINMENT TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/3600:.0f} hours per simulation")
    print(f"Entrainment rate: {ENTRAINMENT_RATE} s^-1 (weak)")
    print(f"Cloud shading: OFF (isolate dynamic entrainment effect)")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_dynamic_entrainment"
    base_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = Path(__file__).parent.parent / "cloud_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Backup config
    config_path = Path(__file__).parent.parent / "config.py"
    config_backup = config_path.parent / "config.py.backup_dyn_entr"
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

            modify_config(test['enable_dynamic'])
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

    if len(results) == 2:
        const = results[0]
        dyn = results[1]

        print(f"\n{'Metric':<20} {'Constant':<15} {'Dynamic':<15} {'Change':<15}")
        print("-" * 70)
        print(f"{'QC max (g/kg)':<20} {const['qc_max']:<15.2f} {dyn['qc_max']:<15.2f} {dyn['qc_max']-const['qc_max']:+.2f}")
        print(f"{'QC mean (g/kg)':<20} {const['qc_mean']:<15.4f} {dyn['qc_mean']:<15.4f} {dyn['qc_mean']-const['qc_mean']:+.4f}")
        print(f"{'Cloud depth (m)':<20} {const['depth']:<15.0f} {dyn['depth']:<15.0f} {dyn['depth']-const['depth']:+.0f}")
        print(f"{'Coverage (%)':<20} {const['cloud_frac']:<15.1f} {dyn['cloud_frac']:<15.1f} {dyn['cloud_frac']-const['cloud_frac']:+.1f}")

        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("="*70)
        if abs(dyn['qc_max'] - const['qc_max']) > 0.5:
            print("✓ Dynamic entrainment has significant effect!")
            print(f"  Cloud water content changed by {(dyn['qc_max']/const['qc_max']-1)*100:.0f}%")
            if dyn['qc_max'] < const['qc_max']:
                print("  → Stronger entrainment during active growth phases")
            else:
                print("  → Weaker entrainment allows more cloud water buildup")
        else:
            print("✗ Effect is small - may need stronger base rate or longer simulation")

    print(f"\n{'='*70}")
    print(f"Results saved to: {base_dir}")
    print(f"GIFs saved to: {gif_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
