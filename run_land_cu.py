"""land_cu runner, kept for the documented commands: same as `run_case.py land_cu ...`.

Usage: python run_land_cu.py [T_END_seconds] [KEY=VALUE ...] [TAG=name]   (default 7200)
"""
import os
import runpy
import sys

sys.argv[1:1] = ['land_cu']
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_case.py'),
               run_name='__main__')
