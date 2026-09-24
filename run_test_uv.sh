#!/bin/bash
# Quick test run to generate snapshots with u,v fields

# Run from experiments/test_uv_save directory
cd experiments/test_uv_save

# Link to restart file
ln -sf ../../snapshots/snap_0001200.npz .

# Run 20 minutes (1200s) with output every 60s
python3 ../../main.py

