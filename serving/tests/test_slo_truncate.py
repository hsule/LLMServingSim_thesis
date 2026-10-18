"""Adversarial property test for serving/core/slo_truncate.py.

Invariant under test: whenever SloTruncator.check(now) returns True, the verdict a complete run reaches over
the same window (linear-interpolation p99 per metric, `analyze_fs_cap.pct`) is infeasible.

Each case is a random synthetic run driven exactly as the simulator drives the truncator:
- requests are created (on_arrival) no later than the first check at or after their arrival, and some are
  "released" late, as closed-loop sub-requests are (their creation happens when a predecessor finishes);
- events (first token, completion) are recorded at a non-decreasing clock, never before arrival;
- check(now) is called at every clock step.

The generator deliberately produces arrivals on the window boundaries, simultaneous arrivals, events exactly
at arrival + SLO, unfinished requests and runs that end exactly at the threshold.
Run: python3 -m serving.tests.test_slo_truncate [cases]
"""
import random
import sys

from serving.core.slo_truncate import SloTruncator, NS


def p99(v):
    v = sorted(v); k = (len(v) - 1) * 0.99; lo = int(k); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def one_case(rnd):
    slo = {"TTFT": rnd.choice([0.5, 1.0, 2.0]), "latency": rnd.choice([3.0, 5.0])}
    warm, win = 1.0, rnd.choice([5.0, 10.0])
    t0 = rnd.randrange(0, 3) * NS
    lo, hi = t0 + int(warm * NS), t0 + int((warm + win) * NS)
    n = rnd.randrange(30, 400)
    heavy_p = rnd.choice([0.0, 0.003, 0.006, 0.01, 0.015, 0.03])    # per case: from clearly feasible to clearly not
    reqs = []
    for i in range(n):
        a = t0 + rnd.randrange(0, int((warm + win + 3) * NS), max(1, int(rnd.choice([1, 1e6, 1e8]))))
        if rnd.random() < 0.05: a = rnd.choice([lo, hi, lo - 1, hi - 1])          # boundary arrivals
        heavy = rnd.random() < heavy_p
        ttft = int(rnd.uniform(0, 3 * slo["TTFT"] if heavy else 0.8 * slo["TTFT"]) * NS)
        if rnd.random() < 0.03: ttft = int(slo["TTFT"] * NS)                       # exactly at the SLO
        lat = ttft + int(rnd.uniform(0, (2 if heavy else 0.4) * slo["latency"]) * NS)
        late = rnd.random() < 0.3                                                   # closed-loop style creation
        create = a + (rnd.randrange(0, NS // 20) if late else 0)
        # a request is created (routed) before it can be scheduled, so no event precedes its creation
        ttft = max(ttft, create - a); lat = max(lat, ttft)
        reqs.append(dict(id=i, a=a, ttft=ttft, lat=lat, create=create))
    t = SloTruncator(slo, warm, win, t0, n)
    times = sorted({r["create"] for r in reqs} | {r["a"] + r["ttft"] for r in reqs} | {r["a"] + r["lat"] for r in reqs}
                   | {r["a"] + int(s * NS) + d for r in reqs for s in slo.values() for d in (0, 1)})
    created = set(); fired_at = None
    for now in times:
        for r in reqs:                                   # creation before events at the same instant
            if r["id"] not in created and r["create"] <= now:
                t.on_arrival(r["id"], r["a"]); created.add(r["id"])
        for r in reqs:
            if r["id"] in created and r["a"] + r["ttft"] == now: t.on_event("TTFT", r["id"], r["ttft"])
            if r["id"] in created and r["a"] + r["lat"] == now: t.on_event("latency", r["id"], r["lat"])
        if t.check(now) and fired_at is None: fired_at = now
    kept = [r for r in reqs if lo <= r["a"] < hi]
    if len(kept) < 2: return None, fired_at
    feasible = all(p99([r["ttft" if m == "TTFT" else "lat"] / NS for r in kept]) <= s for m, s in slo.items())
    return feasible, fired_at


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rnd = random.Random(20261019)
    fired = infeasible = bad = 0
    for _ in range(cases):
        feasible, fired_at = one_case(rnd)
        if feasible is None: continue
        infeasible += not feasible
        if fired_at is not None:
            fired += 1
            if feasible: bad += 1
    print(f"{cases} cases: {infeasible} infeasible, {fired} truncated, {bad} truncated-but-feasible")
    assert bad == 0, "truncation fired on a run whose full verdict is feasible"
    assert fired > 0, "the generator never exercised a truncation"


if __name__ == "__main__":
    main()
