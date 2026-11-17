"""
Unattended layered_day campaign (docs/PLAN_layered_day2.md, 2026-11-04): runs the job queue below stage by stage
(1 cloud LW radiation → 2 falling cloud ice → 3 main 128² runs → 4 resolution test), two jobs side by side
(2 threads each) or one alone (4 threads). After each run: mp4 from the 1-minute snapshots, then thinning to 5 min
(scripts/finish_run.py). State in outputs/logs/campaign_ld2/state.json, so a restart resumes where it stopped.

Run from the repo root:
    ./venv/bin/python3 scripts/campaign_layered_day2.py start     # launch in the background (systemd user unit)
    ./venv/bin/python3 scripts/campaign_layered_day2.py status    # progress table
    ./venv/bin/python3 scripts/campaign_layered_day2.py run       # the scheduler itself, in the foreground
    systemctl --user stop ld2-campaign                            # stop everything (rerun `start` to resume)

`start` again after a stop, crash or reboot resumes: finished jobs are kept; jobs that were running are restarted
from t = 0 (the precursor has no restart support) and the snapshots of their unfinished run are deleted.
A job that fails is logged and skipped; the others still run. Stage n+1 starts only when stage n is over.
No job is started if its estimated end (scaled by the speed measured so far) is past the deadline
(--hours, default 96 h after the first start); such jobs are marked "skipped".
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGDIR = os.path.join(ROOT, 'outputs', 'logs', 'campaign_ld2')
STATE = os.path.join(LOGDIR, 'state.json')
UNIT = 'ld2-campaign'
PY = os.path.join(ROOT, 'venv', 'bin', 'python3')

LW = ["RAD_SCHEME='cloud_lw'"]
ICE = ["ICE_SEDIMENT=True"]
SMALL = ["NX=64", "NY=64", "PRECURSOR_NX=32", "PRECURSOR_NY=32"]      # 9.6 km pilots, 4.8 km precursor tile

# id, stage, variant, T_END [s], overrides, size class, short description (mp4 name), estimated wall hours
# (64² alone 15 wall-min per simulated hour (2026-11-04 check), pair ~25; 128² pair ~120, alone ~80 (2026-11-01)).
# 8 h: the turb layers saturate only after ~4 h (3 cm/s ascent; the 4-h check had no mid/upper cloud yet)
JOBS = [
    # Stage 1 — cloud-top LW cooling / cloud-base warming
    ("R0", 1, "trough",   28800, SMALL,                                 "s", "trough64_ctrl",       3.5),
    ("R1", 1, "trough",   28800, SMALL + LW,                            "s", "trough64_lw",         3.5),
    ("R2", 1, "turb",     28800, SMALL,                                 "s", "turb64_ctrl",         3.5),
    ("R3", 1, "turb",     28800, SMALL + LW,                            "s", "turb64_lw",           3.5),
    ("R4", 1, "turb",     28800, SMALL + LW + ["RAD_F0=100.0", "RAD_F1_EPS=0.5"], "s", "turb64_lwstrong", 3.5),
    ("R5", 1, "turb_jet", 28800, SMALL + LW,                            "s", "turbjet64_lw",        3.5),
    # Stage 2 — falling cloud ice (with and without radiation)
    ("I1", 2, "trough",   28800, SMALL + LW + ICE,                      "s", "trough64_lw_icefall", 3.5),
    ("I2", 2, "turb",     28800, SMALL + ICE,                           "s", "turb64_icefall",      3.5),
    ("I3", 2, "turb",     28800, SMALL + LW + ICE,                      "s", "turb64_lw_icefall",   3.5),
    ("I4", 2, "turb_jet", 28800, SMALL + LW + ICE,                      "s", "turbjet64_lw_icefall", 3.5),
    # Stage 3 — main runs, 128² at 150 m (19.2 km) + 64² precursor, as the first layered_day runs
    ("M1", 3, "turb",     28800, LW + ICE,                              "L", "turb128_lw_icefall",  16.0),
    ("M2", 3, "turb_jet", 28800, LW + ICE,                              "L", "turbjet128_lw_icefall", 16.0),
    ("M3", 3, "trough",   21600, LW + ICE,                              "L", "trough128_lw_icefall", 8.0),
    # Stage 4 — resolution: 100 m, 128² (12.8 km), 64² precursor
    ("H1", 4, "turb",     28800, LW + ICE + ["DX=100.0", "DY=100.0"],   "L", "turb128dx100_lw_icefall", 14.0),
]
DISK_GB = {"s": 6.0, "L": 30.0}          # peak snapshot volume of one run before thinning (+10 GB margin kept)


def now() -> float:
    return time.time()


def stamp(t=None) -> str:
    return dt.datetime.fromtimestamp(t or now()).strftime('%Y-%m-%d %H:%M')


def log(msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGDIR, 'campaign.log'), 'a') as f:
        f.write(line + '\n')


def load() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"started": None, "deadline": None, "jobs": {j[0]: {"status": "pending"} for j in JOBS}}


def save(st: dict) -> None:
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def case_name(job) -> str:
    jid, _, variant, t_end, *_ = job
    return f"layered_day_{variant}_{t_end // 60}min_ld2{jid}"


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def speed_factor(st: dict) -> float:
    """Measured / estimated wall time over the finished jobs (1 before any)."""
    est = {j[0]: j[7] for j in JOBS}
    done = [(v["wall_h"], est[k]) for k, v in st["jobs"].items() if v["status"] == "done" and "wall_h" in v]
    return max(1.0, sum(a for a, _ in done) / sum(b for _, b in done)) if done else 1.0


def launch(job, threads: int) -> subprocess.Popen:
    jid, _, variant, t_end, over, *_ = job
    cmd = [PY, 'run_case.py', f'layered_day:{variant}', str(t_end), 'SNAPSHOT_EVERY=60.0', *over, f'TAG=ld2{jid}']
    env = dict(os.environ, MPLBACKEND='Agg', NUMBA_NUM_THREADS=str(threads))
    out = open(os.path.join(LOGDIR, f'{jid}.runner.log'), 'w')
    log(f"{jid}: start ({threads} threads): {' '.join(cmd[1:])}")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


def run_dir_of(job, t0: float):
    dirs = [d for d in glob.glob(os.path.join(ROOT, 'runs', f'*_{case_name(job)}')) if os.path.getmtime(d) >= t0 - 60]
    return max(dirs, key=os.path.getmtime) if dirs else None


def finish(job, st: dict, rc: int) -> None:
    jid, *_, desc, _ = job
    rec = st["jobs"][jid]
    rec["end"] = now()
    rec["wall_h"] = round((rec["end"] - rec["start"]) / 3600.0, 2)
    logf = os.path.join(LOGDIR, f'{jid}.runner.log')
    ok = rc == 0 and 'RUN COMPLETED' in open(logf, errors='replace').read()
    rd = run_dir_of(job, rec["start"])
    rec["run_dir"] = os.path.relpath(rd, ROOT) if rd else None
    if rd:
        shutil.copy(logf, os.path.join(rd, 'runner.log'))
    if not ok:
        rec["status"] = "failed"
        log(f"{jid}: FAILED (exit {rc}) after {rec['wall_h']} h — see {os.path.relpath(logf, ROOT)}")
        return
    day = os.path.basename(rd)[:8]
    mp4 = f"cloud_gifs/{day}_ld2_{jid}_{desc}_0-{job[3] // 60}min.mp4"
    env = dict(os.environ, VTK_DEFAULT_OPENGL_WINDOW='vtkEGLRenderWindow', NUMBA_NUM_THREADS='2')
    with open(os.path.join(LOGDIR, f'{jid}.finish.log'), 'w') as f:
        rc2 = subprocess.call([PY, 'scripts/finish_run.py', rd, '--out', mp4, '--jobs', '2'],
                              cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    rec["mp4"] = mp4 if rc2 == 0 else None
    rec["status"] = "done" if rc2 == 0 else "done (mp4 failed)"
    log(f"{jid}: done in {rec['wall_h']} h → {rec['run_dir']}" + (f", {mp4}" if rc2 == 0 else ", mp4 FAILED"))


def run(hours: float) -> None:
    os.makedirs(LOGDIR, exist_ok=True)
    st = load()
    if st["started"] is None:
        st["started"] = now()
        st["deadline"] = now() + hours * 3600.0
        log(f"campaign started; deadline {stamp(st['deadline'])}")
    else:
        log(f"campaign resumed; deadline {stamp(st['deadline'])}")
    for jid, rec in st["jobs"].items():                 # interrupted runs: restart from t = 0
        if rec["status"] == "running":
            job = next(j for j in JOBS if j[0] == jid)
            rd = run_dir_of(job, rec["start"])
            if rd and os.path.isdir(os.path.join(rd, 'snapshots')):
                shutil.rmtree(os.path.join(rd, 'snapshots'))
            log(f"{jid}: was interrupted ({rd and os.path.relpath(rd, ROOT)}), snapshots deleted, will restart")
            st["jobs"][jid] = {"status": "pending", "restarts": rec.get("restarts", 0) + 1}
    save(st)

    procs = {}                                           # jid → Popen
    for stage in sorted({j[1] for j in JOBS}):
        log(f"stage {stage}")
        while True:
            for jid, p in list(procs.items()):
                if p.poll() is not None:
                    del procs[jid]
                    finish(next(j for j in JOBS if j[0] == jid), st, p.returncode)
                    save(st)
            pending = [j for j in JOBS if j[1] == stage and st["jobs"][j[0]]["status"] == "pending"]
            if not pending and not procs:
                break
            while pending and len(procs) < 2:
                job = pending[0]
                f = speed_factor(st)
                if now() + job[7] * f * 3600.0 > st["deadline"]:
                    st["jobs"][job[0]]["status"] = "skipped"
                    log(f"{job[0]}: skipped — estimated {job[7] * f:.1f} h would end after the deadline")
                    pending.pop(0)
                    continue
                if free_gb() < DISK_GB[job[5]] * (len(procs) + 1) + 10.0:
                    if procs:
                        break                            # wait for the running job to finish and thin
                    st["jobs"][job[0]]["status"] = "skipped"
                    log(f"{job[0]}: skipped — only {free_gb():.0f} GB free")
                    pending.pop(0)
                    continue
                # two side by side get 2 threads each; a job alone (nothing else left to pair) gets 4
                threads = 2 if (procs or len(pending) > 1) else 4
                procs[job[0]] = launch(job, threads)
                st["jobs"][job[0]].update(status="running", start=now(), threads=threads)
                save(st)
                pending.pop(0)
            time.sleep(30)
    log("campaign finished")
    status()


def status() -> None:
    st = load()
    print(f"started {stamp(st['started']) if st['started'] else '-'}   deadline "
          f"{stamp(st['deadline']) if st['deadline'] else '-'}   speed factor {speed_factor(st):.2f}   "
          f"free disk {free_gb():.0f} GB")
    print(f"{'job':4} {'st':2} {'status':18} {'wall h':>7} {'est h':>6}  run / mp4")
    for j in JOBS:
        r = st["jobs"][j[0]]
        wall = r.get("wall_h", (now() - r["start"]) / 3600.0 if r["status"] == "running" else None)
        extra = r.get("mp4") or r.get("run_dir") or ""
        if r["status"] == "running":
            logf = os.path.join(LOGDIR, f'{j[0]}.runner.log')
            last = [l for l in open(logf, errors='replace') if l.startswith('t=')][-1:] if os.path.exists(logf) else []
            extra = last[0].split('step')[0].strip() + f" of {j[3]}s" if last else extra
        print(f"{j[0]:4} {j[1]:2} {r['status']:18} {wall if wall is None else round(wall, 1)!s:>7} {j[7]:6.1f}  {extra}")


def start(hours: float) -> None:
    os.makedirs(LOGDIR, exist_ok=True)
    active = subprocess.run(['systemctl', '--user', 'is-active', UNIT], capture_output=True, text=True).stdout.strip()
    if active == 'active':
        sys.exit(f"already running (systemctl --user status {UNIT})")
    subprocess.run(['systemctl', '--user', 'reset-failed', UNIT], capture_output=True)
    cmd = ['systemd-run', '--user', f'--unit={UNIT}', f'--working-directory={ROOT}',
           '-p', f'StandardOutput=append:{os.path.join(LOGDIR, "scheduler.out")}',
           '-p', f'StandardError=append:{os.path.join(LOGDIR, "scheduler.out")}',
           PY, os.path.abspath(__file__), 'run', '--hours', str(hours)]
    subprocess.check_call(cmd)
    print(f"started as systemd user unit {UNIT}; progress: {sys.argv[0]} status, "
          f"log: {os.path.relpath(os.path.join(LOGDIR, 'campaign.log'), ROOT)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['start', 'run', 'status'])
    ap.add_argument('--hours', type=float, default=96.0, help='deadline after the first start (default 96)')
    a = ap.parse_args()
    os.chdir(ROOT)
    {'start': lambda: start(a.hours), 'run': lambda: run(a.hours), 'status': status}[a.cmd]()
