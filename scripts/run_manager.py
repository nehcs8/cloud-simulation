#!/usr/bin/env python3
"""
Run management system for cloud simulations.

Handles:
- Automatic directory creation with timestamps
- Metadata YAML generation
- Config archiving
- Lineage tracking (parent -> nested)
"""

import yaml
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class RunManager:
    """Manages simulation runs with metadata and organization."""

    def __init__(self, base_dir: Path = None):
        """
        Parameters
        ----------
        base_dir : Path
            Root directory for simulations (defaults to current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.experiments_dir = self.base_dir / "experiments"
        self.nested_dir = self.base_dir / "nested_runs"

    def create_parent_run(
        self,
        description: str,
        config_dict: Dict[str, Any],
        restart_from: Optional[str] = None,
    ) -> Path:
        """
        Create directory structure for a parent simulation run.

        Parameters
        ----------
        description : str
            Short description of experiment (e.g., "qvsurf_sweep")
        config_dict : dict
            Configuration parameters
        restart_from : str, optional
            Path to restart file if continuing from previous run

        Returns
        -------
        run_dir : Path
            Created run directory
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_id = f"{timestamp}_{description}"
        run_dir = self.experiments_dir / "parent" / run_id

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "snapshots").mkdir(exist_ok=True)

        # Write metadata
        metadata = {
            "run_id": run_id,
            "run_type": "parent",
            "description": description,
            "created": datetime.now().isoformat(),
            "status": "running",
            "restart_from": restart_from,
            "config": config_dict,
        }
        self._write_metadata(run_dir / "metadata.yaml", metadata)

        # Archive config
        self._write_config(run_dir / "config.yaml", config_dict)

        print(f"Created parent run: {run_dir}")
        return run_dir

    def create_nested_run(
        self,
        parent_snapshot: str,
        cloud_info: Dict[str, Any],
        config_dict: Dict[str, Any],
        resolution_m: float,
        duration_s: float,
    ) -> Path:
        """
        Create directory structure for a nested simulation run.

        Parameters
        ----------
        parent_snapshot : str
            Path to parent snapshot used for IC
        cloud_info : dict
            Information about selected cloud:
            - cloud_id: int
            - center: tuple (i, j, k)
            - time_parent: float (seconds)
            - qc_max: float (g/kg)
        config_dict : dict
            Nested configuration parameters
        resolution_m : float
            Grid resolution (meters)
        duration_s : float
            Simulation duration (seconds)

        Returns
        -------
        run_dir : Path
            Created run directory
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        cloud_id = cloud_info.get("cloud_id", "unknown")
        run_id = f"{timestamp}_nest_cloud{cloud_id}_dx{int(resolution_m)}m"
        run_dir = self.nested_dir / run_id

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "snapshots").mkdir(exist_ok=True)

        # Extract parent info
        parent_path = Path(parent_snapshot)
        parent_experiment = self._extract_parent_experiment(parent_path)

        # Write metadata
        metadata = {
            "run_id": run_id,
            "run_type": "nested",
            "created": datetime.now().isoformat(),
            "status": "running",
            "parent_source": {
                "experiment": parent_experiment,
                "snapshot": str(parent_path),
                "time_parent_s": cloud_info.get("time_parent"),
                "cloud_id": cloud_info.get("cloud_id"),
                "cloud_center": cloud_info.get("center"),
                "cloud_qc_max_gkg": cloud_info.get("qc_max"),
            },
            "nested_config": {
                "resolution_m": resolution_m,
                "domain_size_km": [
                    config_dict["NX"] * resolution_m / 1000,
                    config_dict["NY"] * resolution_m / 1000,
                    config_dict["Z_TOP"] / 1000,
                ],
                "grid_cells": [config_dict["NX"], config_dict["NY"], config_dict["NZ"]],
                "duration_s": duration_s,
                "duration_min": duration_s / 60,
                "timestep_s": config_dict["DT"],
                "output_every_s": config_dict["OUTPUT_EVERY"],
            },
            "config": config_dict,
        }
        self._write_metadata(run_dir / "metadata.yaml", metadata)

        # Archive config
        self._write_config(run_dir / "config_nested.yaml", config_dict)

        print(f"Created nested run: {run_dir}")
        return run_dir

    def finalize_run(self, run_dir: Path, status: str = "complete", notes: str = ""):
        """
        Update run metadata when simulation completes.

        Parameters
        ----------
        run_dir : Path
            Run directory
        status : str
            Final status ("complete", "failed", "cancelled")
        notes : str
            Optional notes about the run
        """
        metadata_path = run_dir / "metadata.yaml"
        if not metadata_path.exists():
            print(f"Warning: metadata not found in {run_dir}")
            return

        metadata = self._read_metadata(metadata_path)
        metadata["status"] = status
        metadata["completed"] = datetime.now().isoformat()
        if notes:
            metadata["notes"] = notes

        # Add output statistics if available
        diag_path = run_dir / "diag.csv"
        if diag_path.exists():
            metadata["outputs"] = {
                "diag_csv": str(diag_path),
                "snapshots": str(run_dir / "snapshots"),
                "log_file": str(run_dir / "sim.log"),
            }

        self._write_metadata(metadata_path, metadata)
        print(f"Finalized run {run_dir.name}: {status}")

    def _extract_parent_experiment(self, snapshot_path: Path) -> str:
        """Extract experiment name from snapshot path."""
        parts = snapshot_path.parts
        # Look for experiments/ in path
        try:
            exp_idx = parts.index("experiments")
            # Return everything after experiments/ up to snapshots/
            snap_idx = parts.index("snapshots")
            return "/".join(parts[exp_idx + 1 : snap_idx])
        except ValueError:
            return str(snapshot_path.parent)

    def _write_metadata(self, path: Path, metadata: Dict[str, Any]):
        """Write metadata to YAML file."""
        # Convert tuples to lists for YAML compatibility
        metadata = self._tuples_to_lists(metadata)
        with open(path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    def _tuples_to_lists(self, obj):
        """Recursively convert tuples and numpy types to native Python types for YAML serialization."""
        # Convert numpy types to Python native types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._tuples_to_lists(item) for item in obj]
        else:
            return obj

    def _read_metadata(self, path: Path) -> Dict[str, Any]:
        """Read metadata from YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _write_config(self, path: Path, config_dict: Dict[str, Any]):
        """Write config to YAML file."""
        # Convert config module to dict if needed
        if not isinstance(config_dict, dict):
            config_dict = {
                k: getattr(config_dict, k)
                for k in dir(config_dict)
                if not k.startswith("_") and k.isupper()
            }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def list_runs(self, run_type: str = "all", status: str = "all") -> list:
        """
        List all simulation runs.

        Parameters
        ----------
        run_type : str
            "parent", "nested", or "all"
        status : str
            Filter by status ("running", "complete", "failed", "all")

        Returns
        -------
        runs : list of dict
            List of run metadata
        """
        runs = []

        # Search directories
        search_dirs = []
        if run_type in ["parent", "all"]:
            parent_dir = self.experiments_dir / "parent"
            if parent_dir.exists():
                search_dirs.extend(parent_dir.iterdir())

        if run_type in ["nested", "all"]:
            if self.nested_dir.exists():
                search_dirs.extend(self.nested_dir.iterdir())

        # Read metadata from each run
        for run_dir in search_dirs:
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "metadata.yaml"
            if not metadata_path.exists():
                continue

            metadata = self._read_metadata(metadata_path)
            if status == "all" or metadata.get("status") == status:
                runs.append(metadata)

        return runs


def config_to_dict(config_module) -> Dict[str, Any]:
    """
    Convert config module to dictionary.

    Parameters
    ----------
    config_module : module
        Configuration module (config.py or config_nested.py)

    Returns
    -------
    config_dict : dict
        Dictionary of configuration parameters
    """
    return {
        k: getattr(config_module, k)
        for k in dir(config_module)
        if not k.startswith("_") and k.isupper()
    }


if __name__ == "__main__":
    # Example usage
    manager = RunManager()

    # List all runs
    print("\n=== All Simulation Runs ===")
    runs = manager.list_runs()
    for run in runs:
        print(f"{run['run_id']}: {run['status']} - {run.get('description', 'nested')}")
