#!/usr/bin/env python3
"""
Automated experiment runner for cloud simulation parameter sweep.

Usage:
    python run_experiment.py --exp-id exp01 --phase phase1_qvsurf \\
        --qv-surf 0.011 --duration 1800 --restart snap_0001200.npz
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np


def update_config(params: dict) -> None:
    """Update config.py with experimental parameters."""
    config_path = Path("config.py")
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # Parameters to update
    updates = {
        'QV_SURF': params.get('qv_surf'),
        'QV_SCALE': params.get('qv_scale'),
        'TAU_NUDGE_Q': params.get('tau_nudge_q'),
        'LHF': params.get('lhf'),
        'W_SUBS': params.get('w_subs'),
        'T_END': params.get('t_end'),
        'RESTART_FROM': params.get('restart_from'),
    }

    new_lines = []
    for line in lines:
        modified = False
        for key, value in updates.items():
            if value is not None and line.strip().startswith(f'{key}:'):
                # Handle string vs numeric values
                if key == 'RESTART_FROM':
                    if value == 'None':
                        new_lines.append(f'{key}: str | None = None\n')
                    else:
                        new_lines.append(f'{key}: str | None = "{value}"\n')
                elif isinstance(value, str):
                    new_lines.append(f'{key}: float = {value}\n')
                else:
                    new_lines.append(f'{key}: float = {value}\n')
                modified = True
                break
        if not modified:
            new_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def prepare_restart_file(restart_name: str) -> bool:
    """
    Prepare restart file from safe repository.

    CRITICAL: Always copy from restart_snapshots/ to avoid contamination!
    The snapshots/ directory gets overwritten during experiments.
    """
    if restart_name == 'None':
        return True  # No restart needed

    # Clean snapshots directory to prevent contamination
    snapshots_dir = Path('snapshots')
    if snapshots_dir.exists():
        for old_snap in snapshots_dir.glob('snap_*.npz'):
            old_snap.unlink()
        print(f"   🧹 Cleared {len(list(snapshots_dir.glob('snap_*.npz')))} old snapshots")
    else:
        snapshots_dir.mkdir(parents=True)

    # Copy from safe repository
    restart_repo = Path('restart_snapshots')
    src_file = restart_repo / restart_name
    dst_file = snapshots_dir / restart_name

    if not src_file.exists():
        print(f"   ✗ Restart file not found: {src_file}", file=sys.stderr)
        print(f"   Available restart files:", file=sys.stderr)
        for f in restart_repo.glob('*.npz'):
            print(f"     - {f.name}", file=sys.stderr)
        return False

    # Validate restart file before copying
    try:
        data = np.load(src_file)
        t = float(data['t'])
        qv_checksum = data['qv'].sum()
        qc_checksum = data['qc'].sum()
        print(f"   ✓ Restart validation: t={t:.0f}s, qv_sum={qv_checksum:.3f}, qc_sum={qc_checksum:.3f}")
    except Exception as e:
        print(f"   ✗ Restart file validation failed: {e}", file=sys.stderr)
        return False

    # Copy to working directory
    shutil.copy(src_file, dst_file)
    print(f"   📋 Copied restart: {src_file.name} → snapshots/")

    return True


def run_simulation(exp_dir: Path) -> bool:
    """Run the simulation and return True if successful."""
    log_file = exp_dir / "sim.log"

    try:
        # Run simulation with Agg backend
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'

        result = subprocess.run(
            ['./venv/bin/python3', 'main.py'],
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            env=env,
            timeout=3600  # 1 hour timeout
        )

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Simulation timed out after 1 hour", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}", file=sys.stderr)
        return False


def analyze_results(exp_dir: Path) -> dict:
    """Extract key metrics from diag.csv."""
    diag_path = Path("diag.csv")

    if not diag_path.exists():
        return {
            'cloud_frac_mean': 0.0,
            'cloud_frac_std': 0.0,
            'qc_max_mean': 0.0,
            'n_clouds_mean': 0.0,
            'qv_bl_start': 0.0,
            'qv_bl_end': 0.0,
            'qv_bl_drift': 0.0,
            'status': 'FAILED'
        }

    # Read diagnostics
    with open(diag_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 5:
        return {'status': 'INCOMPLETE'}

    # Skip first few rows (spinup) and analyze rest
    analysis_rows = rows[3:]  # Skip first 3 output times

    cloud_frac = [float(r['cloud_frac']) for r in analysis_rows]
    qc_max = [float(r['qc_max']) for r in analysis_rows]
    n_clouds = [float(r['n_clouds']) for r in analysis_rows]
    qv_bl = [float(r['qv_bl']) * 1000 for r in analysis_rows]  # Convert to g/kg

    qv_bl_start = qv_bl[0] if qv_bl else 0.0
    qv_bl_end = qv_bl[-1] if qv_bl else 0.0
    qv_bl_drift = qv_bl_end - qv_bl_start

    cloud_frac_mean = np.mean(cloud_frac) if cloud_frac else 0.0
    cloud_frac_std = np.std(cloud_frac) if cloud_frac else 0.0
    qc_max_mean = np.mean(qc_max) if qc_max else 0.0
    n_clouds_mean = np.mean(n_clouds) if n_clouds else 0.0

    # Classify result
    if cloud_frac_mean >= 0.10 and cloud_frac_mean <= 0.20 and abs(qv_bl_drift) < 0.5 and qc_max_mean > 0.5:
        status = 'GOOD'
    elif (cloud_frac_mean >= 0.05 and cloud_frac_mean <= 0.25) or abs(qv_bl_drift) < 1.0:
        status = 'MARGINAL'
    else:
        status = 'POOR'

    return {
        'cloud_frac_mean': cloud_frac_mean,
        'cloud_frac_std': cloud_frac_std,
        'qc_max_mean': qc_max_mean,
        'n_clouds_mean': n_clouds_mean,
        'qv_bl_start': qv_bl_start,
        'qv_bl_end': qv_bl_end,
        'qv_bl_drift': qv_bl_drift,
        'status': status
    }


def save_outputs(exp_dir: Path) -> None:
    """Copy key outputs to experiment directory."""
    outputs = ['diag.csv', 'sim_run_qv_fixed.log']

    for output in outputs:
        src = Path(output)
        if src.exists():
            shutil.copy(src, exp_dir / output)

    # Copy ALL snapshots for GIF generation and comparison
    snap_dir = exp_dir / 'snapshots'
    snap_dir.mkdir(exist_ok=True)

    snapshots = sorted(Path('snapshots').glob('snap_*.npz'))
    for snap in snapshots:
        shutil.copy(snap, snap_dir / snap.name)

    print(f"   💾 Saved {len(snapshots)} snapshots for GIF generation")


def append_to_summary(results: dict, summary_file: Path) -> None:
    """Append experiment results to summary CSV."""
    file_exists = summary_file.exists()

    with open(summary_file, 'a', newline='') as f:
        fieldnames = [
            'exp_id', 'phase', 'qv_surf', 'qv_scale', 'tau_nudge_q', 'lhf', 'w_subs',
            'cloud_frac_mean', 'cloud_frac_std', 'qc_max_mean', 'n_clouds_mean',
            'qv_bl_start', 'qv_bl_end', 'qv_bl_drift', 'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)


def main():
    parser = argparse.ArgumentParser(description='Run cloud simulation experiment')
    parser.add_argument('--exp-id', required=True, help='Experiment ID (e.g., exp01)')
    parser.add_argument('--phase', required=True, help='Phase directory (e.g., phase1_qvsurf)')
    parser.add_argument('--qv-surf', type=float, default=None)
    parser.add_argument('--qv-scale', type=float, default=None)
    parser.add_argument('--tau-nudge-q', type=float, default=None)
    parser.add_argument('--lhf', type=float, default=None)
    parser.add_argument('--w-subs', type=float, default=None)
    parser.add_argument('--duration', type=float, required=True, help='Simulation duration (s)')
    parser.add_argument('--restart', type=str, default='None', help='Restart file (or None)')

    args = parser.parse_args()

    # Create experiment directory
    exp_dir = Path('experiments') / args.phase / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔬 Running experiment: {args.exp_id}")
    print(f"   Phase: {args.phase}")
    print(f"   Duration: {args.duration}s")

    # Prepare restart file (BEFORE updating config!)
    if args.restart != 'None':
        print("   📂 Preparing restart file...")
        if not prepare_restart_file(args.restart):
            print(f"   ✗ Failed to prepare restart file")
            return 1

    # Prepare parameters
    params = {
        'qv_surf': args.qv_surf,
        'qv_scale': args.qv_scale,
        'tau_nudge_q': args.tau_nudge_q,
        'lhf': args.lhf,
        'w_subs': args.w_subs,
        't_end': args.duration,
        'restart_from': f'snapshots/{args.restart}' if args.restart != 'None' else None,
    }

    # Update config
    print("   📝 Updating config.py...")
    update_config(params)

    # Run simulation
    print("   ▶ Running simulation...")
    success = run_simulation(exp_dir)

    if not success:
        print(f"   ✗ Experiment {args.exp_id} FAILED")
        return 1

    # Analyze results
    print("   📊 Analyzing results...")
    metrics = analyze_results(exp_dir)

    # Save outputs
    print("   💾 Saving outputs...")
    save_outputs(exp_dir)

    # Compile full results
    results = {
        'exp_id': args.exp_id,
        'phase': args.phase,
        'qv_surf': args.qv_surf or 'N/A',
        'qv_scale': args.qv_scale or 'N/A',
        'tau_nudge_q': args.tau_nudge_q or 'N/A',
        'lhf': args.lhf or 'N/A',
        'w_subs': args.w_subs or 'N/A',
        **metrics
    }

    # Save to summary
    summary_file = Path('experiments') / 'results_summary.csv'
    append_to_summary(results, summary_file)

    # Report
    status_emoji = {'GOOD': '✓', 'MARGINAL': '~', 'POOR': '✗', 'FAILED': '✗', 'INCOMPLETE': '?'}
    emoji = status_emoji.get(metrics['status'], '?')
    print(f"   {emoji} Status: {metrics['status']}")
    print(f"   Cloud cover: {metrics['cloud_frac_mean']*100:.1f}% ± {metrics['cloud_frac_std']*100:.1f}%")
    print(f"   qc_max: {metrics['qc_max_mean']:.2f} g/kg")
    print(f"   qv_bl drift: {metrics['qv_bl_drift']:.2f} g/kg")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
