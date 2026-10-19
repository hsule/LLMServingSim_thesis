"""Opt-in early stop of a run whose p99 SLO verdict is already decided infeasible.

The verdict this mirrors: requests arriving in the window [t0 + warmup, t0 + warmup + window) are kept
(t0 = the run's first arrival, the window test done in seconds as the analysis does it), the run is invalid
with fewer than `min_window_reqs` kept requests, and a valid run meets the SLO iff, for every metric, the
linear-interpolation p99 over the kept requests is <= the SLO. With N kept requests and
m = floor(0.99 (N - 1)), a run is certainly infeasible once a metric has at least N - m kept requests above
its SLO (the p99 then lies between two values above the SLO).

A request is known to exceed a metric's SLO S as soon as simulated time passes arrival + S without the event
that metric measures (first token for TTFT, completion for latency), or as soon as that event is recorded
later than arrival + S. Both facts are final, so the count only grows and stopping on it never changes the
verdict a complete run would reach. N is only known once every arrival is known (closed-loop sub-requests
arrive as earlier ones finish), so the threshold uses an upper bound on N: the kept requests seen so far plus,
while the window is still open, every request whose arrival is not known yet. The threshold N - m is
non-decreasing in N, so the bound makes the stop conservative, never wrong. The stop also waits until
`min_window_reqs` requests are kept: before that the full run could still turn out invalid.

Single runs only. A part of a split trace has its own first arrival and its own request count, so its window
and its threshold are not the whole trace's; there is no split mode until the aggregation over parts is
verified (a reader must treat a truncated part as invalid).
"""
import heapq
import json

NS = 1_000_000_000
METRICS = ("TTFT", "latency")      # the events Request reports (set_ttft, add_latency)
MIN_WINDOW_REQS = 200              # analyze_fs_cap.MIN_WINDOW_REQS


class SloTruncator:
    def __init__(self, slo, warmup_s, window_s, t0_ns, total_requests, min_window_reqs=MIN_WINDOW_REQS):
        bad = [m for m in slo if m not in METRICS]
        if bad or not slo:
            raise ValueError(f"--slo-truncate metrics must be among {METRICS}, got {sorted(slo) or 'none'}")
        if any(s <= 0 for s in slo.values()):
            raise ValueError("--slo-truncate SLOs must be positive")
        if warmup_s < 0 or window_s <= 0:
            raise ValueError("--slo-window needs warmup >= 0 and window > 0")
        self.slo = {m: int(round(s * NS)) for m, s in slo.items()}       # metric -> SLO in ns
        self.t0, self.warmup, self.window = t0_ns, warmup_s, window_s
        self.hi = t0_ns + int(round((warmup_s + window_s) * NS))         # window end, only to tell N is final
        self.total = total_requests
        self.min_window_reqs = min_window_reqs
        self.seen = 0                  # requests whose arrival is known
        self.kept = set()              # ids arriving inside the window
        self.done = {m: set() for m in self.slo}       # ids whose metric event has been recorded
        self.over = {m: set() for m in self.slo}       # ids known to exceed the metric's SLO
        self.heap = []                 # (deadline_ns, metric, id) for kept requests
        self.last_now = None
        self.fired = None

    def in_window(self, arrival):
        d = (arrival - self.t0) / NS                   # the analysis's own test (analyze_fs_cap.read_cell)
        return self.warmup <= d < self.warmup + self.window

    # ---- hooks called from Request -------------------------------------------------------------
    def on_arrival(self, rid, arrival):
        self.seen += 1
        if self.seen > self.total:
            raise RuntimeError(f"--slo-truncate saw {self.seen} requests, more than the {self.total} counted")
        if self.in_window(arrival):
            self.kept.add(rid)
            for m, s in self.slo.items():
                heapq.heappush(self.heap, (arrival + s, m, rid))

    def on_event(self, metric, rid, value):
        if metric not in self.slo or rid not in self.kept:
            return
        self.done[metric].add(rid)
        if value > self.slo[metric]:
            self.over[metric].add(rid)

    # ---- the check ------------------------------------------------------------------------------
    def threshold(self, now):
        unknown = self.total - self.seen
        n_upper = len(self.kept) + (unknown if now < self.hi else 0)
        if n_upper <= 0:
            return None
        return n_upper - int((n_upper - 1) * 0.99)

    def check(self, now):
        """True once some metric is certainly over its p99 SLO. Idempotent after it fires."""
        if self.fired:
            return True
        if self.last_now is not None and now < self.last_now:
            # every "no event by now" violation relies on events being recorded at a non-decreasing clock
            raise RuntimeError(f"--slo-truncate clock went backwards: {now} < {self.last_now}")
        self.last_now = now
        while self.heap and self.heap[0][0] < now:
            _, m, rid = heapq.heappop(self.heap)
            if rid not in self.done[m]:
                self.over[m].add(rid)
        thr = self.threshold(now)
        if thr is None or len(self.kept) < self.min_window_reqs:
            return False
        for m in self.slo:
            v = len(self.over[m])
            if v >= thr:
                self.fired = dict(verdict="INFEASIBLE", scope="single-run", metric=m, violations=v,
                                  threshold=thr, kept_so_far=len(self.kept), arrivals_known=self.seen,
                                  total_requests=self.total, min_window_reqs=self.min_window_reqs,
                                  t_ns=int(now), t0_ns=self.t0, warmup_s=self.warmup, window_s=self.window,
                                  slo_ns=self.slo)
                return True
        return False

    def write(self, path):
        with open(path, "w") as f:
            json.dump(self.fired, f, indent=1)


_ACTIVE = None


def install(truncator):
    global _ACTIVE
    _ACTIVE = truncator


def active():
    return _ACTIVE
