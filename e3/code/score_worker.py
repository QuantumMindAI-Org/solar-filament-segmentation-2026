"""TRACK2-R3 scoring worker: a persistent fresh interpreter, started ONCE at the beginning of the run (while the
training process is still small) by track2.ScoringPool as

    python -B <src_root>/track2_train_r3b/score_worker.py SRC_ROOT INBOX_DIR PROGRESS_FILE

Why at the start: on Linux, ru_maxrss survives execve (the kernel records the old address space's peak into the new
image), so an interpreter forked or vforked from the 4-5 GiB training process would begin with that peak and the
pinned scorer's 2 GiB guard would refuse in the child. Started early, the worker inherits at most the small
parent's peak. It reports its own ru_maxrss at start (`start_maxrss`) so the smoke log shows the fact.

A plain module file run by path: no multiprocessing, no re-import of the parent's __main__, so it starts the same way
whether the kernel ran as a script, through runpy, or as exec'd notebook text. Importing it has no side effects.
It imports the project from SRC_ROOT and then serves jobs: INBOX_DIR/job_NNNNNN.pkl (pickled validation components:
GT frame, uint8 maps, disks, settings, soft codes), in sequence. It writes JSON lines to PROGRESS_FILE: ready,
job_start, task_start, task_done | error, job_done, exit. A cancel_NNNNNN file skips the job's remaining tasks; a
STOP file ends the worker; it also ends if its parent dies. The parent enforces every deadline; this file only
reports. It reads no dataset file.
Test faults (dry runs only; the parent strips the variable otherwise): QMAI_TRACK2_TEST_FAULT = import | kill | sleep.
"""
import json
import os
import resource
import signal
import sys
import time


def maxrss_bytes():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform == 'darwin' else v * 1024)


def main(argv):
    start_maxrss = maxrss_bytes()          # before any import: what this interpreter inherited at exec
    t0 = time.time()
    parent = os.getppid()
    src_root, inbox, progress_path = argv[1:4]
    out = open(progress_path, 'a', buffering=1)

    def emit(**kw):
        out.write(json.dumps(kw) + '\n')
        out.flush()
    fault = os.environ.get('QMAI_TRACK2_TEST_FAULT', '')
    if fault == 'import':
        raise ImportError('QMAI_TEST_FAULT_IMPORT: injected start-up failure')
    sys.path.insert(0, os.path.join(src_root, 'track2_train_r3b'))
    import pickle
    import track2
    proj = track2.get_project(src_root)
    emit(event='ready', pid=os.getpid(), import_s=time.time() - t0, start_maxrss=start_maxrss,
         ready_maxrss=maxrss_bytes(), fork_sentinel=getattr(track2, '_FORK_SENTINEL', None),
         argv0=os.path.basename(sys.argv[0]))
    seq = 0
    while True:
        if os.path.exists(os.path.join(inbox, 'STOP')) or os.getppid() != parent:
            emit(event='exit')
            return 0
        job_path = os.path.join(inbox, 'job_%06d.pkl' % seq)
        if not os.path.exists(job_path):
            time.sleep(0.05)
            continue
        with open(job_path, 'rb') as f:
            job = pickle.load(f)
        soft = tuple(job['soft'])
        emit(event='job_start', seq=seq)
        for task in job['tasks']:
            if os.path.exists(os.path.join(inbox, 'cancel_%06d' % seq)):
                break
            ci = task[0]
            t1 = time.time()
            emit(event='task_start', seq=seq, ci=ci)
            if fault == 'kill':
                os.kill(os.getpid(), signal.SIGKILL)
            if fault == 'sleep':
                time.sleep(3600)
            try:
                res, refused = track2.score_component(proj, *task[1:], soft=soft)
            except Exception as e:  # noqa: BLE001  the parent stops the run with this name
                emit(event='error', seq=seq, ci=ci, error='%s: %s' % (type(e).__name__, e))
                return 3
            emit(event='task_done', seq=seq, ci=ci, pid=os.getpid(), peak=maxrss_bytes(), seconds=time.time() - t1,
                 refused=refused, res=[[limb, [list(s) for s in s3], rows.tolist()] for limb, s3, rows in res])
        os.unlink(job_path)
        emit(event='job_done', seq=seq)
        seq += 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
