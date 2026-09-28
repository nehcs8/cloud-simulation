#!/usr/bin/env python3
"""
Test autoconversion threshold for fluffy cumulus clouds.

Problem: Clouds are small and evaporate quickly because autoconversion
         converts cloud water to rain too aggressively.

Solution: Increase autoconversion threshold (QC0) to let clouds accumulate
          water before raining out.

Test matrix:
1. QC0 = 2e-4 kg/kg (0.2 g/kg) - BASELINE (current aggressive conversion)
2. QC0 = 1e-3 kg/kg (1.0 g/kg) - MODERATE (let clouds grow bigger)
3. QC0 = 2e-3 kg/kg (2.0 g/kg) - HIGH (fluffy non-precipitating cumulus)
4. Autoconversion OFF - EXTREME (no rain, pure cloud water accumulation)

All tests use:
- Dynamic entrainment: ON (0.001 s^-1, velocity-dependent)
- Cloud shading: ON (extinction=200, min_trans=0.1) - strongest from previous test
- Duration: 2 hours each
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        'name': 'baseline_qc0_0.2',
        'qc0': 2e-4,
        'enable_autoconv': True,
        'desc': 'Baseline: QC0=0.2 g/kg (aggressive conversion)'
    },
    {
        'name': 'moderate_qc0_1.0',
        'qc0': 1e-3,
        'enable_autoconv': True,
        'desc': 'Moderate: QC0=1.0 g/kg (let clouds grow)'
    },
    {
        'name': 'high_qc0_2.0',
        'qc0': 2e-3,
        'enable_autoconv': True,
        'desc': 'High: QC0=2.0 g/kg (fluffy non-precip cumulus)'
    },
    {
        'name': 'no_autoconversion',
        'qc0': 1e-3,  # unused
        'enable_autoconv': False,
        'desc': 'Extreme: Autoconversion OFF (no rain production)'
    },
]

SIM_DURATION = 7200.0  # 2 hours
OUTPUT_EVERY = 60.0
DX = 100.0

def modify_config(qc0, enable_autoconv):
    """Modify config.py for test."""
    config_path = Path(__file__).parent.parent / "config.py"

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        # Enable both dynamic entrainment and cloud shading (best combo from previous tests)
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

    print(f"✓ Config: QC0={qc0*1e3:.1f} g/kg, autoconv={enable_autoconv}")

def modify_microphysics(qc0, enable_autoconv):
    """Modify microphysics.py to change autoconversion parameters."""
    micro_path = Path(__file__).parent.parent / "microphysics.py"

    with open(micro_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('_K1    ='):
            if enable_autoconv:
                modified_lines.append(f"_K1    = 1e-3    # s^-1,  autoconversion rate\n")
            else:
                modified_lines.append(f"_K1    = 0.0     # s^-1,  autoconversion DISABLED\n")
        elif line.startswith('_QC0   ='):
            modified_lines.append(f"_QC0   = {qc0}    # kg/kg, autoconversion threshold\n")
        else:
            modified_lines.append(line)

    with open(micro_path, 'w') as f:
        f.writelines(modified_lines)

    if enable_autoconv:
        print(f"✓ Microphysics: QC0 = {qc0:.1e} kg/kg ({qc0*1e3:.1f} g/kg)")
    else:
        print(f"✓ Microphysics: Autoconversion DISABLED")

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

        # Get time series for analysis
        qc_max_series = []
        qr_max_series = []
        cloud_frac_series = []
        for line in lines[1:]:
            data = line.strip().split(',')
            row = dict(zip(header, data))
            try:
                qc_max_series.append(float(row.get('qc_max', 0)))
                qr_max_series.append(float(row.get('qr_max', 0)))
                cloud_frac_series.append(float(row.get('cloud_frac', 0)))
            except ValueError:
                pass

    import numpy as np

    try:
        # Time-averaged metrics
        qc_max_mean = np.mean(qc_max_series)
        qc_max_std = np.std(qc_max_series)
        qr_max_mean = np.mean(qr_max_series)
        cloud_frac_mean = np.mean(cloud_frac_series) * 100

        return {
            'qc_max_mean': qc_max_mean,
            'qc_max_std': qc_max_std,
            'qr_max_mean': qr_max_mean,
            'cloud_frac_mean': cloud_frac_mean,
        }
    except (ValueError, KeyError):
        return None

def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("AUTOCONVERSION THRESHOLD TEST")
    print("="*70)
    print(f"Duration: {SIM_DURATION/3600:.0f} hours per simulation")
    print(f"Total tests: {len(TESTS)}")
    print(f"\nGoal: Find QC0 that creates fluffy, persistent cumulus clouds")
    print("="*70 + "\n")

    date_prefix = datetime.now().strftime("%Y%m%d")
    base_dir = Path(__file__).parent.parent / "experiments" / f"{date_prefix}_autoconversion"
    base_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = Path(__file__).parent.parent / "cloud_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Backup files
    config_path = Path(__file__).parent.parent / "config.py"
    micro_path = Path(__file__).parent.parent / "microphysics.py"
    config_backup = config_path.parent / "config.py.backup_autoconv"
    micro_backup = micro_path.parent / "microphysics.py.backup_autoconv"
    shutil.copy(config_path, config_backup)
    shutil.copy(micro_path, micro_backup)
    print(f"✓ Backed up config and microphysics\n")

    results = []

    try:
        for i, test in enumerate(TESTS, 1):
            test_name = test['name']
            output_dir = base_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f"# TEST {i}/{len(TESTS)}: {test['desc']}")
            print(f"{'#'*70}\n")

            modify_config(test['qc0'], test['enable_autoconv'])
            modify_microphysics(test['qc0'], test['enable_autoconv'])
            success = run_simulation(test_name, output_dir)

            if success:
                move_outputs(output_dir)

                # Generate GIF
                gif_path = gif_dir / f"{date_prefix}_autoconv_{test_name}.gif"
                generate_gif(output_dir / "snapshots", gif_path)

                metrics = extract_metrics(output_dir)
                if metrics:
                    results.append({
                        'name': test['name'],
                        'qc0': test['qc0'],
                        'desc': test['desc'],
                        **metrics
                    })

    finally:
        # Restore files
        shutil.copy(config_backup, config_path)
        shutil.copy(micro_backup, micro_path)
        config_backup.unlink()
        micro_backup.unlink()
        print(f"\n✓ Restored original files")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    if results:
        print(f"\n{'QC0 (g/kg)':<15} {'QC mean±std':<20} {'QR mean':<15} {'CloudFrac':<12}")
        print("-" * 70)

        for r in results:
            qc0_str = f"{r['qc0']*1e3:.1f}" if r.get('qc0', 0) > 0 else "OFF"
            qc_str = f"{r['qc_max_mean']:.2f}±{r['qc_max_std']:.2f}"
            qr_str = f"{r['qr_max_mean']:.2f}"
            cf_str = f"{r['cloud_frac_mean']:.1f}%"
            print(f"{qc0_str:<15} {qc_str:<20} {qr_str:<15} {cf_str:<12}")

        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("="*70)
        print("\nCloud water accumulation (QC mean):")
        print("  - Higher QC → clouds hold more water → bigger, fluffier clouds")
        print("  - Lower QR → less rain production → clouds persist longer")
        print("\nOscillation strength (QC std):")
        print("  - Higher std → stronger pulsation (growing/shrinking cycles)")

        max_qc = max(r['qc_max_mean'] for r in results)
        best = max(results, key=lambda r: r['qc_max_mean'])

        print(f"\n✓ Best cloud accumulation: {best['desc']}")
        print(f"  QC max: {best['qc_max_mean']:.2f} g/kg")
        print(f"  Rain reduction: {(1 - best['qr_max_mean']/results[0]['qr_max_mean'])*100:.0f}%")

    print(f"\n{'='*70}")
    print(f"Results saved to: {base_dir}")
    print(f"GIFs saved to: {gif_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
