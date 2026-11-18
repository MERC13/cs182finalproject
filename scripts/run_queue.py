"""Run icl.train jobs from a jobs file, P at a time; skips runs already done.

Each non-empty line of the jobs file is the argument string for icl.train
(must contain --out).  Restart-safe: rerunning resumes the unfinished jobs.

  python scripts/run_queue.py jobs/sweep.txt --procs 4
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def system_load():
    """Number of python training processes on the machine (any queue or manual launch)."""
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0]).startswith(b"python") and b"-m" in argv:
            mod = argv[argv.index(b"-m") + 1] if argv.index(b"-m") + 1 < len(argv) else b""
            n += mod in (b"icl.train", b"icl.linattn")
    return n


def done(out):
    try:
        return json.load(open(out))["status"] != "running"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    todo = []
    for j in jobs:
        args = shlex.split(j)
        out = args[args.index("--out") + 1]
        if not done(os.path.join(ROOT, out)):
            todo.append((args, out))
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.procs and system_load() < a.procs:
            # jobs that resume from a checkpoint wait until that checkpoint exists (dependency)
            ready = [i for i, (ar, _) in enumerate(todo)
                     if "--resume" not in ar or os.path.exists(os.path.join(ROOT, ar[ar.index("--resume") + 1]))]
            if not ready:
                break
            args, out = todo.pop(ready[0])
            os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
            # an interrupted run (e.g. a container restart) continues from its own last checkpoint
            own = out.replace(".json", ".ckpt")
            if "--resume" not in args and os.path.exists(os.path.join(ROOT, own)):
                args = [*args, "--resume", own]
            log = open(os.path.join(ROOT, out.replace(".json", ".log")), "a")
            p = subprocess.Popen([sys.executable, "-m", "icl.train", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, out, log))
            print(f"[{time.time()-t0:7.0f}s] start {out}", flush=True)
            time.sleep(3)  # let the child exec so system_load() sees it
        time.sleep(5)
        for item in running[:]:
            p, out, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                try:
                    L = json.load(open(os.path.join(ROOT, out)))
                    msg = f"{L['status']} val={L.get('final_val_loss')} wall={L.get('wall')}"
                except Exception as e:
                    msg = f"exit {p.returncode} ({e})"
                print(f"[{time.time()-t0:7.0f}s] end   {out}: {msg}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
