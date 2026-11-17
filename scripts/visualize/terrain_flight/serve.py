"""Serve the terrain flight viewer on http://127.0.0.1:8765 (local only).

    ./venv/bin/python3 scripts/visualize/terrain_flight/serve.py [--port 8765] [--open]

/            -> this folder (index.html, flight.js)
/tiles/...   -> data/dem/glo30_lod/  (built by build_tiles.py)
"""
import argparse
import functools
import glob
import http.server
import json
import os
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'dem', 'glo30_lod'))
SUN_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'sun'))
SUN_DIR = None   # --sun; default: the finest finished data/sun/*/viewer (re-checked on every request)


def sun_dir():
    if SUN_DIR:
        return SUN_DIR
    best = None
    for mj in glob.glob(os.path.join(SUN_ROOT, '*', 'viewer', 'meta.json')):
        try:
            res = json.load(open(mj))['res_m']
        except (OSError, ValueError, KeyError):
            continue
        if best is None or res < best[0]:
            best = (res, os.path.dirname(mj))
    return best[1] if best else os.path.join(SUN_ROOT, '__none__')


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        p = path.split('?', 1)[0].split('#', 1)[0]
        for prefix, base in (('/tiles/', TILES), ('/sun/', None)):
            if p.startswith(prefix):
                base = base or sun_dir()
                rel = os.path.normpath(p[len(prefix):]).lstrip('/')
                if rel.startswith('..'):
                    return os.path.join(base, '__nope__')
                return os.path.join(base, rel)
        return super().translate_path(path)

    def end_headers(self):
        if self.path.startswith('/tiles/'):
            self.send_header('Cache-Control', 'max-age=86400')
        else:
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, *a):   # quiet: thousands of tile requests
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--open', action='store_true', help='open the browser')
    ap.add_argument('--sun', help='sun overlay folder (data/sun/<run>/viewer); default: finest available')
    a = ap.parse_args()
    global SUN_DIR
    SUN_DIR = os.path.abspath(a.sun) if a.sun else None
    if not os.path.exists(os.path.join(TILES, 'meta.json')):
        raise SystemExit(f"No tiles in {TILES}: run build_tiles.py first")
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), functools.partial(Handler, directory=HERE))
    url = f'http://127.0.0.1:{a.port}/'
    print(f"terrain flight viewer at {url}  (Ctrl+C to stop); sun overlay from {sun_dir()}", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
