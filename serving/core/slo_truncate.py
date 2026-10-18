"""Opt-in early stop of a run whose p99 SLO verdict is already decided infeasible.

The verdict this mirrors: requests arriving in the window [t0 + warmup, t0 + warmup + window) are kept, and
the run meets the SLO iff, for every metric, the linear-interpolation p99 over the kept requests is <= the
SLO. With N kept requests and m = floor(0.99 (N - 1)), a run is certainly infeasible once a metric has at
least N - m kept requests above its SLO (the p99 then lies between two values above the SLO).

A request is known to exceed a metric's SLO S as soon as simulated time passes arrival + S without the event
that metric measures (first token for TTFT, completion for latency), or as soon as that event is recorded
later than arrival + S. Both facts are final, so the count only grows and stopping on it never changes the
verdict a complete run would reach. N is only known once every arrival is known (closed-loop sub-requests
arrive as earlier ones finish), so the threshold uses an upper bound on N: the kept requests seen so far plus,
while the window is still open, every request whose arrival is not known yet. The threshold N - m is
non-decreasing in N, so the bound makes the stop conservative, never wrong.

`prior` carries violations already established for the same window by other parts of the same trace
(a split replayed part by part); `margin` widens every SLO by a timing tolerance for such parts.
"""
import heapq
import json

NS = 1_000_000_000


class SloTruncator:
    def __init__(self, slo, warmup_s, window_s, t0_ns, total_requests, prior=None, margin_s=0.0):
        self.slo = {m: int(round(s * NS)) for m, s in slo.items()}       # metric -> SLO in ns
        self.margin = int(round(margin_s * NS))
        self.lo = t0_ns + int(round(warmup_s * NS))
        self.hi = self.lo + int(round(window_s * NS))
        self.prior = dict(prior or {})
        self.total = total_requests
        self.seen = 0                  # requests whose arrival is known
        self.kept = set()              # ids arriving inside the window
        self.done = {m: set() for m in self.slo}       # ids whose metric event has been recorded
        self.over = {m: set() for m in self.slo}       # ids known to exceed the metric's SLO
        self.heap = []                 # (deadline_ns, metric, id) for kept requests
        self.fired = None

    # ---- hooks called from Request -------------------------------------------------------------
    def on_arrival(self, rid, arrival):
        self.seen += 1
        if self.lo <= arrival < self.hi:
            self.kept.add(rid)
            for m, s in self.slo.items():
                heapq.heappush(self.heap, (arrival + s + self.margin, m, rid))

    def on_event(self, metric, rid, value):
        if metric not in self.slo or rid not in self.kept:
            return
        self.done[metric].add(rid)
        if value > self.slo[metric] + self.margin:
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
        while self.heap and self.heap[0][0] < now:
            _, m, rid = heapq.heappop(self.heap)
            if rid not in self.done[m]:
                self.over[m].add(rid)
        thr = self.threshold(now)
        if thr is None:
            return False
        for m in self.slo:
            v = len(self.over[m]) + self.prior.get(m, 0)
            if v >= thr:
                self.fired = dict(verdict="INFEASIBLE", metric=m, violations=len(self.over[m]),
                                  prior=self.prior.get(m, 0), threshold=thr, kept_so_far=len(self.kept),
                                  arrivals_known=self.seen, total_requests=self.total, t_ns=int(now),
                                  window_ns=[self.lo, self.hi], slo_ns=self.slo, margin_ns=self.margin)
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
