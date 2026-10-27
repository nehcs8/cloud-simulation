"""
Re-encode a rendered GIF without re-rendering: write an .mp4 (H.264) next to it, and optionally rewrite
the GIF at a new frame rate. Every frame is kept, so it works on runs whose snapshots are already thinned.

Run: ./venv/bin/python3 scripts/visualize/reencode.py cloud_gifs/X.gif [Y.gif ...] [--fps 3] [--no-gif]
Needs imageio-ffmpeg (bundled ffmpeg binary) in the venv.
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np


def reencode(path, fps, gif=True):
    frames = [np.asarray(f)[..., :3] for f in imageio.mimread(path, memtest=False)]
    h, w = frames[0].shape[:2]
    frames = [f[:h - h % 2, :w - w % 2] for f in frames]          # H.264 needs even dimensions
    mp4 = os.path.splitext(path)[0] + '.mp4'
    imageio.mimsave(mp4, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1, pixelformat='yuv420p')
    if gif:
        imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"{path}: {len(frames)} frames at {fps:g} fps → {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB)"
          + (" + GIF rewritten" if gif else ""))
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gifs', nargs='+')
    ap.add_argument('--fps', type=float, default=3.0)
    ap.add_argument('--no-gif', action='store_true', help='only write the mp4, leave the GIF as it is')
    a = ap.parse_args()
    for g in a.gifs:
        reencode(g, a.fps, gif=not a.no_gif)


if __name__ == '__main__':
    main()
