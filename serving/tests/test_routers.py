"""Behaviour tests for the request-aware routing policies in serving/core/router.py.

Drives Router directly with stub schedulers: requests are routed with route_arrived_requests(t) and
completed with notify_request_completed(id, t), and each policy's documented rule is checked against an
independent restatement of it. No simulator run.
Run: python3 -m serving.tests.test_routers
"""
import bisect
import hashlib
import json
import os
import random
import shutil
import tempfile

from serving.core.router import Router

MS = 1_000_000


class StubScheduler:
    def __init__(self, i):
        self.instance_id, self.pd_type, self.model, self.enable_prefix_caching = i, None, "m", False
        self.waiting, self.running, self.max_num_seqs, self.got = [], [], 128, []

    def add_request(self, req, is_init=True):
        self.got.append(req[0])


def make(policy, n, rows, **kw):
    scheds = [StubScheduler(i) for i in range(n)]
    r = Router(n, scheds, 0, policy, **kw)
    root = tempfile.mkdtemp(); os.mkdir(os.path.join(root, "cwd"))
    with open(os.path.join(root, "trace.jsonl"), "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    cwd = os.getcwd(); os.chdir(os.path.join(root, "cwd"))   # load_requests reads ../<path>, as from astra-sim/
    try:
        r.load_requests("trace.jsonl")
    finally:
        os.chdir(cwd); shutil.rmtree(root)
    return r, scheds


def route_one(r):
    """Route the next pending request at its own arrival time; return (request id, instance)."""
    req = r._pending_requests[r._pending_idx]
    r.route_arrived_requests(req['arrival_time_ns'])
    return req['index'], r._req_instance[req['index']]


def H(key): return int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:12], 16)


def flat(n, gap=MS, toks=None, prio=None):
    rnd = random.Random(1)
    return [dict(input_toks=toks[i] if toks else rnd.randrange(1, 4000), output_toks=10, arrival_time_ns=(i + 1) * gap,
                 **({} if prio is None else {"priority": prio[i]})) for i in range(n)]


def test_inflight_tok():
    toks = [100, 50, 10, 500, 20, 20]
    r, _ = make("INFLIGHT_TOK", 3, flat(6, toks=toks))
    load = [0, 0, 0]
    for k in range(6):
        rid, inst = route_one(r)
        assert inst == min(range(3), key=lambda i: load[i]), (k, inst, load)
        load[inst] += toks[rid]
        if k == 3:                                         # complete request 0: its tokens leave
            r.notify_request_completed(0, 4 * MS); load[r_inst0] -= toks[0]
        if k == 0: r_inst0 = inst


def test_inflight_lag():
    r, _ = make("INFLIGHT_LAG", 2, flat(6, gap=60 * MS), stats_interval_ns=100 * MS)
    seen = []
    # arrivals at 60, 120, 180, ... ms; request 0 completes at 70 ms, visible from the 100 ms poll on
    rid, inst = route_one(r); seen.append(inst)                                  # t=60: tie -> 0
    r.notify_request_completed(rid, 70 * MS)
    rid, inst = route_one(r); seen.append(inst)                                  # t=120: completion seen, tie -> 0
    rid, inst = route_one(r); seen.append(inst)                                  # t=180: 0 has 1, 1 has 0 -> 1
    assert seen == [0, 0, 1], seen
    r2, _ = make("INFLIGHT_LAG", 2, flat(3, gap=30 * MS), stats_interval_ns=100 * MS)
    rid, inst = route_one(r2)                                                    # t=30 -> 0
    r2.notify_request_completed(rid, 40 * MS)
    rid, inst = route_one(r2)                                                    # t=60: completion not yet polled
    assert inst == 1, "a completion became visible before the next poll"


def test_priority():
    prio = [1, 0, 1, 0, 1, 1, 0]
    r, _ = make("PRIORITY", 3, flat(7, prio=prio), priority_threshold=1)
    rr, inflight, got = 0, [0, 0, 0], []
    for k in range(7):
        rid, inst = route_one(r)
        if prio[rid] < 1: want = min(range(3), key=lambda i: inflight[i])
        else: want = rr % 3; rr += 1
        assert inst == want, (k, inst, want); inflight[inst] += 1


def test_hashring_qps():
    rows = []
    for i in range(40):
        row = dict(input_toks=10, output_toks=5, arrival_time_ns=(i + 1) * 100 * MS)
        if i % 3 == 0: row = dict(session_id=f"s{i}", arrival_time_ns=row["arrival_time_ns"],
                                  sub_requests=[dict(input_toks=10, output_toks=5, tool_duration_ns=0)])
        rows.append(row)
    r, _ = make("HASHRING_QPS", 4, rows, qps_window_ns=1000 * MS, hash_vnodes=50)
    ring = sorted((H(f"{i}-{v}"), i) for i in range(4) for v in range(50))
    recent = [[] for _ in range(4)]
    for k in range(40):
        req = r._pending_requests[r._pending_idx]; a = req['arrival_time_ns']
        rid, inst = route_one(r)
        recent = [[x for x in q if x > a - 1000 * MS] for q in recent]
        if 'session_id' in req:
            j = bisect.bisect_left(ring, (H(req['session_id']), -1)); want = ring[j % len(ring)][1]
        else:
            want = min(range(4), key=lambda i: len(recent[i]))
        assert inst == want, (k, inst, want); recent[inst].append(a)


def test_hash_load():
    rows = [dict(session_id="same", arrival_time_ns=(i + 1) * MS,
                 sub_requests=[dict(input_toks=10, output_toks=5, tool_duration_ns=0)]) for i in range(6)]
    r, _ = make("HASH_LOAD", 3, rows, load_threshold=2)
    h = H("same") % 3
    got = [route_one(r)[1] for _ in range(6)]
    # the hashed instance takes requests until it holds 2, then the least loaded (ties lowest index) do
    inflight, want = [0, 0, 0], []
    for _ in range(6):
        i = h if inflight[h] < 2 else min(range(3), key=lambda j: inflight[j])
        want.append(i); inflight[i] += 1
    assert got == want, (got, want)


def test_p2c3():
    r, _ = make("P2C3", 5, flat(30))
    rnd = random.Random(42); inflight = [0] * 5
    for k in range(30):
        cands = []
        while len(cands) < 3:
            c = rnd.randrange(5)
            if c not in cands: cands.append(c)
        want = min(cands, key=lambda i: inflight[i])
        rid, inst = route_one(r)
        assert inst == want, (k, inst, want); inflight[inst] += 1
        if k % 4 == 3:
            r.notify_request_completed(rid, (k + 1) * MS); inflight[inst] -= 1


def main():
    for t in (test_inflight_tok, test_inflight_lag, test_priority, test_hashring_qps, test_hash_load, test_p2c3):
        t(); print(f"{t.__name__}: ok")


if __name__ == "__main__":
    main()
