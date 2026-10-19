"""Adversarial property test for serving/core/slo_truncate.py.

Invariant under test: whenever SloTruncator.check(now) returns True, the verdict a complete run reaches over
the same window is "valid and infeasible": t0 = first arrival, window [warmup, warmup + window) s after t0
tested in seconds, at least `min_window_reqs` kept requests, linear-interpolation p99 per metric (the same
function as the analysis's `pct`).

Each case is a random synthetic run driven the way the simulator drives the truncator:
- every request is created (on_arrival) at a check at or after its arrival. A check at time `now` may still
  miss requests arriving exactly at `now`: a closed-loop sub-request released with zero tool time inside the
  ask is routed one tick later, never earlier than its arrival and never later than the next check after it;
- events (first token, completion) are recorded at a non-decreasing clock, never before creation;
- check(now) is called at every clock step, and sometimes also before that step's events are recorded, as
  when several NPUs report at the same tick.

The generator produces arrivals on the window boundaries, simultaneous arrivals, events exactly at arrival +
SLO, windows of fractional seconds, runs below the minimum window size, and runs that end exactly at the
threshold. The rejection tests cover the input checks (metric names, window, clock order, request count).
Run: python3 -m serving.tests.test_slo_truncate [cases]
"""
import random
import sys

from serving.core.slo_truncate import SloTruncator, NS


def pct(xs, p):            # analyze_fs_cap.pct
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def one_case(rnd):
    slo = rnd.choice([{"TTFT": rnd.choice([0.5, 1.0, 2.0])}, {"latency": rnd.choice([3.0, 5.0])},
                      {"TTFT": rnd.choice([0.5, 1.0, 2.0]), "latency": rnd.choice([3.0, 5.0])}])
    warm, win = rnd.choice([(1.0, 5.0), (1.0, 10.0), (0.1, 0.2), (0.3, 4.7)])
    min_reqs = rnd.choice([2, 50, 200])
    n = rnd.randrange(30, 500)
    heavy_p = rnd.choice([0.0, 0.003, 0.006, 0.01, 0.015, 0.03])    # per case: from clearly feasible to clearly not
    base = rnd.randrange(0, 3) * NS
    span = int((warm + win + 3) * NS)
    reqs = []
    for i in range(n):
        a = base + rnd.randrange(0, span, max(1, int(rnd.choice([1, 1e6, 1e8]))))
        reqs.append(dict(id=i, a=a))
    t0 = min(r["a"] for r in reqs)
    lo, hi = t0 + int(round(warm * NS)), t0 + int(round((warm + win) * NS))
    for r in reqs:
        if rnd.random() < 0.05 and r["a"] != t0:
            r["a"] = rnd.choice([lo, hi, lo - 1, hi - 1, lo + 1, hi + 1])               # boundary arrivals
    t0 = min(r["a"] for r in reqs)
    for r in reqs:
        a = r["a"]
        heavy = rnd.random() < heavy_p
        ttft = int(rnd.uniform(0, 3 * 2.0 if heavy else 0.8 * 0.5) * NS)
        if rnd.random() < 0.03: ttft = int(slo.get("TTFT", 1.0) * NS)                  # exactly at the SLO
        lat = ttft + int(rnd.uniform(0, 10.0 if heavy else 1.0) * NS)
        if rnd.random() < 0.03 and "latency" in slo: lat = max(ttft, int(slo["latency"] * NS))
        # zero-tool closed-loop release: routed at the next tick, after any check at the arrival instant
        create = a + 1 if rnd.random() < 0.3 else a
        ttft = max(ttft, create - a); lat = max(lat, ttft)
        r.update(ttft=ttft, lat=lat, create=create)
    t = SloTruncator(slo, warm, win, t0, n, min_window_reqs=min_reqs)
    times = sorted({r["a"] for r in reqs} | {r["create"] for r in reqs} | {r["a"] + r["ttft"] for r in reqs}
                   | {r["a"] + r["lat"] for r in reqs}
                   | {r["a"] + int(s * NS) + d for r in reqs for s in slo.values() for d in (0, 1)} | {hi, hi + 1})
    created = set(); fired_at = None
    for now in times:
        for r in reqs:
            if r["id"] not in created and r["create"] <= now:
                t.on_arrival(r["id"], r["a"]); created.add(r["id"])
        if rnd.random() < 0.5 and t.check(now) and fired_at is None:
            fired_at = now               # another NPU's ask at the same tick, before this one's events
        for r in reqs:
            if r["id"] in created and r["a"] + r["ttft"] == now: t.on_event("TTFT", r["id"], r["ttft"])
            if r["id"] in created and r["a"] + r["lat"] == now: t.on_event("latency", r["id"], r["lat"])
        if t.check(now) and fired_at is None: fired_at = now
    kept = [r for r in reqs if warm <= (r["a"] - t0) / NS < warm + win]
    if len(kept) < min_reqs:
        return "invalid", fired_at
    ok = all(pct([r["ttft" if m == "TTFT" else "lat"] / NS for r in kept], 99) <= s for m, s in slo.items())
    return ("feasible" if ok else "infeasible"), fired_at


def expect_error(fn, what):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError(f"accepted {what}")


def rejection_tests():
    expect_error(lambda: SloTruncator({"ttft": 5}, 30, 270, 0, 10), "a metric name the simulator never reports")
    expect_error(lambda: SloTruncator({"compl_p99": 30}, 30, 270, 0, 10), "an analysis column name")
    expect_error(lambda: SloTruncator({}, 30, 270, 0, 10), "no metric")
    expect_error(lambda: SloTruncator({"TTFT": 0}, 30, 270, 0, 10), "a zero SLO")
    expect_error(lambda: SloTruncator({"TTFT": 5}, 0, 0, 0, 10), "an empty window")
    t = SloTruncator({"TTFT": 5}, 30, 270, 0, 1)
    t.on_arrival(0, 0)
    expect_error(lambda: t.on_arrival(1, 0), "more requests than counted")
    t = SloTruncator({"TTFT": 5}, 30, 270, 0, 10)
    t.check(10)
    expect_error(lambda: t.check(9), "a clock going backwards")
    # below the minimum window size nothing fires, however many violations
    t = SloTruncator({"TTFT": 1}, 0, 10, 0, 5, min_window_reqs=200)
    for i in range(5): t.on_arrival(i, 0)
    assert not t.check(100 * NS), "fired on a run the analysis would call invalid"


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rejection_tests()
    rnd = random.Random(20261020)
    count = {"feasible": 0, "infeasible": 0, "invalid": 0}; fired = bad = 0
    for _ in range(cases):
        verdict, fired_at = one_case(rnd)
        count[verdict] += 1
        if fired_at is not None:
            fired += 1
            if verdict != "infeasible": bad += 1
    print(f"{cases} cases: {count['feasible']} feasible, {count['infeasible']} infeasible, {count['invalid']} invalid; "
          f"{fired} truncated, {bad} truncated but not infeasible")
    assert bad == 0, "truncation fired on a run whose full verdict is not 'valid and infeasible'"
    assert fired > 0, "the generator never exercised a truncation"


if __name__ == "__main__":
    main()
