"""
Finish a run: render its GIF from the full (1-minute) snapshots, write the same frames as an .mp4 next to it,
then thin the snapshots.

The GIF is always made from every snapshot (one frame each, normally 60 s apart). Only after the GIF is written
and has one frame per snapshot are the snapshots thinned to multiples of --keep-every seconds (default 300 s,
which scripts/compare_microphysics.py and the other analysis scripts use). Nothing is deleted if rendering fails.

Run (from the repo root):
    VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py RUN_DIR \\
        --out cloud_gifs/YYYYMMDD_<description>.gif [--keep-every 300] [--jobs 4] [--no-thin] [renderer options …]
Extra options after the known ones (e.g. --zexp 1.5, --xrange 0 6.4) are passed to render_clouds.py; the
defaults here are --fps 3 --smooth 0 --zexp 1.0 (single_cu towers in proportion; 2 fps before 2026-10-27).
The mp4 (H.264, same frames and fps) comes from scripts/visualize/reencode.py; needs imageio-ffmpeg.
"""
import argparse
import glob
import os
import subprocess
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize'))
from reencode import reencode

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-every', type=float, default=300.0)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--no-thin', action='store_true')
    a, extra = ap.parse_known_args()

    snaps = sorted(glob.glob(os.path.join(a.run_dir, 'snapshots', 'snap_*.npz')))
    times = [float(os.path.basename(f)[5:-4]) for f in snaps]
    if len(times) > 1 and min(b - a_ for a_, b in zip(times, times[1:])) > 60.0 + 1e-6:
        print(f"warning: snapshots are {times[1] - times[0]:.0f} s apart (not 1-minute data)")

    opts = {'--fps': '3', '--smooth': '0', '--zexp': '1.0'}
    for i, x in enumerate(extra):
        if x in opts and i + 1 < len(extra):
            opts.pop(x)
    cmd = [sys.executable, os.path.join(HERE, 'visualize', 'render_clouds.py'), a.run_dir, '--out', a.out,
           '--jobs', str(a.jobs)] + [v for kv in opts.items() for v in kv] + extra
    print(' '.join(cmd))
    if subprocess.call(cmd) != 0 or not os.path.exists(a.out):
        sys.exit("rendering failed — snapshots kept")
    n = len(imageio.mimread(a.out, memtest=False))
    print(f"{a.out}: {n} frames for {len(snaps)} snapshots")
    if n != len(snaps):
        sys.exit("frame count differs from the snapshot count — snapshots kept")
    fps = float(extra[extra.index('--fps') + 1]) if '--fps' in extra else float(opts['--fps'])
    reencode(a.out, fps, gif=False)
    if a.no_thin:
        return
    drop = [f for f, t in zip(snaps, times) if t % a.keep_every > 1e-6]
    for f in drop:
        os.remove(f)
    print(f"thinned: removed {len(drop)} snapshots, kept {len(snaps) - len(drop)} (every {a.keep_every:.0f} s)")


if __name__ == '__main__':
    main()
