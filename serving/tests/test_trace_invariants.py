"""Regression tests over real emitted traces.

Run from the repo root inside the simulator container::

    python3 -m serving.tests.test_trace_invariants

The load-bearing one is COMP_NODE duration. ASTRA-Sim does not reject a
zero-duration COMP_NODE; it silently never fires it, the dependents wait
forever, and the run stops advancing with no traceback. The process then spins
until it is killed, which shows up as exit code 137 and is easily misread as an
OOM. A whole 24-cell grid was lost to that once -- see the note in
``trace_generator._emit_layer``.

These use ``--save-trace-text`` to get the human-readable trace, then assert on
its rows. That is the only place the simulator's actual emission can be
inspected without re-implementing it.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

REPO = "/app/LLMServingSim"
FAILURES: list[str] = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL {msg}")
    return cond


def run_sim(cluster, dataset, nreq, run_id, outdir):
    """Run one short simulation with trace text saved; return the trace dir."""
    cmd = [
        "python3", "-m", "serving",
        "--cluster-config", cluster,
        "--dtype", "bfloat16", "--block-size", "16",
        "--dataset", dataset, "--num-reqs", str(nreq),
        "--output", os.path.join(outdir, f"{run_id}.csv"),
        "--run-id", run_id,
        "--log-level", "WARNING",
        "--save-trace-text",
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    check(r.returncode == 0, f"{run_id}: simulation exited {r.returncode}")
    return os.path.join(REPO, "astra-sim", "inputs", "runs", run_id)


def read_traces(root):
    """Yield (path, header_lines, rows) for every model trace under root.

    Only files under ``trace/`` are model traces. ``event_handler.txt`` at the
    run root is a one-line control trace and carries no model layers.
    """
    for p in sorted(glob.glob(os.path.join(root, "trace", "**", "*.txt"),
                              recursive=True)):
        with open(p) as f:
            lines = [ln.rstrip("\n") for ln in f]
        if len(lines) < 4:
            continue
        # line 0: COLOCATED ...; line 1: num_layers; line 2: column header
        body = [ln.split() for ln in lines[3:] if ln.strip()]
        yield p, lines[:3], body


def test_no_zero_duration(root, label):
    """The invariant: no emitted COMP_NODE may have comp_time == 0."""
    total = bad = files = 0
    worst = []
    for p, _, rows in read_traces(root):
        files += 1
        for r in rows:
            if len(r) < 2 or r[0] in ("PIM", "EXPERT"):
                continue
            try:
                t = int(r[1])
            except ValueError:
                continue
            total += 1
            if t <= 0:
                bad += 1
                if len(worst) < 5:
                    worst.append(f"{os.path.basename(p)}: {r[0]} comp_time={t}")
    print(f"  {label}: {files} trace files, {total} COMP_NODE rows, "
          f"{bad} with comp_time <= 0")
    for w in worst:
        print(f"    {w}")
    check(files > 0, f"{label}: no trace files found")
    check(bad == 0, f"{label}: {bad} COMP_NODE rows have comp_time <= 0")


def layer_of(name):
    """``qk_norm_17`` -> ``qk_norm``. Trace rows carry a per-row index suffix."""
    head, _, tail = name.rpartition("_")
    return head if head and tail.isdigit() else name


def test_olmoe_allgather(root):
    """OLMoE emits exactly two ALLGATHERs per qk_norm, sized by q and k."""
    print("  OLMoE qk_norm ALLGATHER semantics:")
    seen_pairs = 0
    checked = False
    for p, _, rows in read_traces(root):
        qk = [r for r in rows if r and layer_of(r[0]) == "qk_norm"]
        if not qk:
            continue
        ag = [r for r in qk if len(r) > 8 and r[8].startswith("ALLGATHER")]
        check(len(ag) == len(qk),
              f"{os.path.basename(p)}: {len(qk)} qk_norm rows but "
              f"{len(ag)} carry ALLGATHER")
        check(len(qk) % 2 == 0,
              f"{os.path.basename(p)}: {len(qk)} qk_norm rows is odd; "
              f"the q/k pair must always be emitted together")
        # Rows come in q,k pairs. Both carry a non-zero comm_size, and for
        # OLMoE num_heads == num_kv_heads so the two sizes are equal. The
        # second row of each pair is the compute-free one: its comp_time is
        # the 1 ns floor, never 0 (see the deadlock note in _emit_layer).
        for i in range(0, len(ag) - 1, 2):
            a, b = int(ag[i][9]), int(ag[i + 1][9])
            ta, tb = int(ag[i][1]), int(ag[i + 1][1])
            check(a > 0 and b > 0,
                  f"{os.path.basename(p)}: ALLGATHER comm_size 0 ({a}, {b})")
            check(a == b,
                  f"{os.path.basename(p)}: q/k ALLGATHER sizes differ "
                  f"({a} vs {b}); OLMoE has num_heads == num_kv_heads")
            check(ta > 0 and tb > 0,
                  f"{os.path.basename(p)}: qk_norm pair comp_time "
                  f"({ta}, {tb}) must both be > 0")
            check(tb <= ta,
                  f"{os.path.basename(p)}: the second row of the pair should "
                  f"carry no kernel time ({tb} vs {ta})")
            seen_pairs += 1
        checked = True
        break
    check(checked, "no qk_norm rows found in the OLMoE trace")
    print(f"    {seen_pairs} q/k ALLGATHER pairs, all non-zero and equal-sized")


def main():
    tmp = tempfile.mkdtemp(prefix="trace_inv_")
    print("=== OLMoE, tp2 (exercises the synthetic ALLGATHER path) ===")
    root = run_sim("configs/cluster/_wd/dp4_tp2_ep2_fast.json",
                   "workloads/_wd/tiny.jsonl", 2, "inv_olmoe_tp2", tmp)
    test_no_zero_duration(root, "olmoe-tp2")
    test_olmoe_allgather(root)

    print("\n=== Llama-3.1-8B, tp2 (no tp_collective declared) ===")
    root = run_sim("configs/cluster/single_node_tp_pp_instance.json",
                   "workloads/example_trace.jsonl", 4, "inv_llama_tp2", tmp)
    test_no_zero_duration(root, "llama-tp2")
    for p, _, rows in read_traces(root):
        # EXPERT / PIM marker lines legitimately carry their own collectives;
        # only real layer rows are covered by tp_collective.
        ag = [r for r in rows
              if r and r[0] not in ("EXPERT", "PIM")
              and len(r) > 8 and r[8].startswith("ALLGATHER")]
        check(not ag, f"{os.path.basename(p)}: Llama emitted {len(ag)} layer-row "
                      f"ALLGATHERs; only families declaring tp_collective should")
        break
    print("    no layer-row ALLGATHERs, as expected without tp_collective")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL TRACE INVARIANTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
