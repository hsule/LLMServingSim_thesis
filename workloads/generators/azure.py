"""Azure LLM inference trace (2024 release) → LLMServingSim JSONL.

Ingests one of the two 2024 one-week CSVs published in
``Azure/AzurePublicDataset`` (release tag ``dataset-llm-2024``)::

    TIMESTAMP,ContextTokens,GeneratedTokens
    2024-05-10 00:00:00.009930+00:00,2162,5

(the 2023 sample in the same repository writes ``...46.6805900`` with a
.NET 7-digit fraction and no zone; both shapes are accepted, and the
2024 files are CRLF)

The CSVs are large (code ≈ 692 MB, conversation ≈ 1.13 GB) and are not
part of this repository. Download them yourself and pass the local path
with ``--source``; the file is streamed, and the scan stops as soon as
the requested window is full.

Ingestion follows the frozen rules of the FleetSieve moving-oracle
pre-registration (``thesis-agentic-sharding/fleetsieve-moving-oracle-audit.md``
§5.2 and §5.3), and implements nothing beyond them:

* **one contiguous window per trace**, starting at that trace's *own*
  ``first_timestamp + 48 h``. The two files do not cover the same week, so
  there is no common wall-clock offset; each file is anchored to itself.
  The window is the first ``--num-reqs`` rows at or after that instant;
* ``input_toks := ContextTokens`` and ``output_toks := GeneratedTokens``,
  copied unmodified;
* ``arrival_time_ns`` from ``TIMESTAMP`` deltas relative to the first row
  of the window, rescaled by **one** scalar (``--scale``, the arrival-rate
  multiplier) and by nothing else. The first request arrives at 0;
* **no filtering by length, no resampling, no truncation of outliers.**
  Rows the simulator may not like (zero generated tokens, for instance)
  are emitted unchanged and only *counted* in the summary;
* ``--scale`` is restricted to the frozen ladder
  ``{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}``. An off-ladder value is an
  error, not a widening: §5.3 says to report and stop instead.

No token ids are produced. The Azure traces have no prompt content (the
publishers redact it for GDPR), so there is nothing to tokenize, and
Stage A is simulator-only. ``request_block_hashes`` treats a request with
no token ids as prefix-cache-disabled while still allocating normally, so
these workloads run under either prefix-caching setting; they simply
never register a cache hit.

Usage::

    python -m workloads.generators azure \\
        --source /path/to/AzureLLMInferenceTrace_conv_1week.csv \\
        --scale 1/16 \\
        --output workloads/_fs/azure-conv-100-s16.jsonl

A ``.manifest.json`` is written next to the output recording the frozen
parameters, the resolved window and the achieved rates, so a cell can be
traced back to the exact rows it replayed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

# Frozen in §5.3. Every entry is a negative power of two, so both "1/16"
# and "0.0625" parse to exactly the same float and exact comparison is safe.
SCALE_LADDER: tuple[float, ...] = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)

# Frozen in §5.2: each trace is anchored to its own first timestamp.
WINDOW_OFFSET_HOURS = 48.0

# Frozen in §5.1/§5.3: 8 cells of 100 requests.
NUM_REQS = 100

EXPECTED_HEADER = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_TS_PREFIX_LEN = 19  # len("2024-05-10 00:00:00")

_NS_PER_SEC = 1_000_000_000
_NS_PER_HOUR = 3_600 * _NS_PER_SEC


# ---------------------------------------------------------------------------
# CLI plumbing — invoked from workloads.generators.__main__
# ---------------------------------------------------------------------------

def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True,
                   help="Local path to AzureLLMInferenceTrace_{code,conv}_1week.csv "
                        "(2024 release). Streamed, never loaded whole.")
    p.add_argument("--scale", required=True,
                   help="Arrival-rate multiplier relative to the trace's native "
                        "rate. Must be on the frozen ladder: "
                        "1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1 "
                        "(decimals such as 0.0625 are accepted).")
    p.add_argument("--output", required=True,
                   help="Output JSONL path.")
    p.add_argument("--num-reqs", type=int, default=NUM_REQS,
                   dest="num_reqs",
                   help=f"Requests to emit. Frozen at {NUM_REQS}; overriding "
                        f"this departs from the pre-registration and is "
                        f"recorded in the manifest.")
    p.add_argument("--window-offset-hours", type=float,
                   default=WINDOW_OFFSET_HOURS,
                   dest="window_offset_hours",
                   help=f"Hours after the trace's own first timestamp at which "
                        f"the window starts. Frozen at {WINDOW_OFFSET_HOURS:g}; "
                        f"overriding this departs from the pre-registration and "
                        f"is recorded in the manifest.")
    p.add_argument("--manifest", default=None,
                   help="Manifest path. Default: <output>.manifest.json. "
                        "Pass 'none' to skip writing it.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    scale = _parse_scale(args.scale)
    if args.num_reqs <= 0:
        raise ValueError(f"--num-reqs must be positive, got {args.num_reqs}")
    if args.window_offset_hours < 0:
        raise ValueError(
            f"--window-offset-hours must be non-negative, got {args.window_offset_hours}")

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(
            f"Azure trace not found: {src}\n"
            f"Download it from the 2024 release:\n"
            f"  https://github.com/Azure/AzurePublicDataset/releases/download/"
            f"dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv\n"
            f"  https://github.com/Azure/AzurePublicDataset/releases/download/"
            f"dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv")

    window = _read_window(src, args.window_offset_hours, args.num_reqs)
    rows = window["rows"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_ns = rows[0][0]
    arrivals: list[int] = []
    with out_path.open("w", encoding="utf-8") as fout:
        for ts_ns, ctx_toks, gen_toks in rows:
            # The single permitted transform: delta from the window's first
            # row, divided by the rate multiplier. Nothing else.
            arrival_ns = round((ts_ns - base_ns) / scale)
            if arrival_ns < 0:
                raise ValueError(
                    f"Negative arrival at row {len(arrivals)}: the window's first "
                    f"row is not its earliest timestamp, so the source is not "
                    f"sorted in the selected range. Refusing to reorder it, "
                    f"because §5.2 forbids resampling.")
            arrivals.append(arrival_ns)
            fout.write(json.dumps({
                "input_toks": ctx_toks,
                "output_toks": gen_toks,
                "arrival_time_ns": arrival_ns,
            }) + "\n")

    summary = _summarize(window, arrivals, scale)
    _report(src, out_path, args, scale, summary)

    if (args.manifest or "").lower() != "none":
        manifest_path = Path(args.manifest) if args.manifest \
            else out_path.with_suffix(out_path.suffix + ".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(_manifest(src, out_path, args, scale, summary), indent=2) + "\n",
            encoding="utf-8")
        print(f"Wrote manifest -> {manifest_path}")

    print(f"Wrote {len(rows)} requests -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Scale ladder
# ---------------------------------------------------------------------------

def _parse_scale(text: str) -> float:
    """Parse ``--scale`` and enforce the frozen ladder.

    Accepts ``1/16`` and ``0.0625`` alike. Off-ladder values are rejected:
    §5.3 says an inadmissible workload is reported, not rescued by widening
    the ladder.
    """
    raw = str(text).strip()
    try:
        value = float(Fraction(raw))
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"--scale is not a number: {text!r}") from e
    if value not in SCALE_LADDER:
        ladder = ", ".join(f"1/{round(1 / s)}" if s != 1.0 else "1"
                           for s in SCALE_LADDER)
        raise ValueError(
            f"--scale {raw} is off the frozen ladder ({ladder}). §5.3: if no "
            f"ladder step is admissible for a workload, report and stop rather "
            f"than widening the ladder.")
    return value


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def _split_timestamp(s: str) -> tuple[str, str, str]:
    """Split a timestamp into (whole seconds, fractional digits, zone suffix).

    The two published generations differ, so both are handled:

    * 2023 sample: ``2023-11-16 18:15:46.6805900`` — .NET 7-digit fraction,
      no zone;
    * 2024 one-week release: ``2024-05-10 00:00:00.009930+00:00`` —
      6-digit fraction and an explicit UTC offset.

    Either part may be absent. The zone is searched for from the right and
    never past the whole-seconds prefix, so the date's own ``-`` separators
    cannot be mistaken for a negative offset.
    """
    s = s.strip()
    zone = ""
    if s[-1:] in ("Z", "z"):
        zone, s = "Z", s[:-1]
    else:
        for i in range(len(s) - 1, _TS_PREFIX_LEN - 1, -1):
            if s[i] in "+-":
                zone, s = s[i:], s[:i]
                break
    head, _, frac = s.partition(".")
    return head, frac, zone


def _zone_offset_ns(zone: str) -> int:
    """``+00:00`` / ``-0730`` / ``Z`` / ``""`` -> offset from UTC in ns."""
    if not zone or zone in ("Z", "z"):
        return 0
    body = zone[1:].replace(":", "")
    if len(body) not in (2, 4) or not body.isdigit():
        raise ValueError(f"Unrecognised timezone suffix in timestamp: {zone!r}")
    minutes = int(body[:2]) * 60 + (int(body[2:4]) if len(body) == 4 else 0)
    return (-1 if zone[0] == "-" else 1) * minutes * 60 * _NS_PER_SEC


def _parse_timestamp_ns(s: str) -> tuple[int, int]:
    """Timestamp text -> ``(epoch nanoseconds, zone offset in ns)``.

    ``datetime.fromisoformat`` rejects the 2023 file's 7-digit fraction, so
    the fraction is taken by hand. A timestamp with no zone is read as UTC,
    so the result never depends on the host's timezone.
    """
    head, frac, zone = _split_timestamp(s)
    dt = datetime.strptime(head, _TS_FMT).replace(tzinfo=timezone.utc)
    ns = int(dt.timestamp()) * _NS_PER_SEC
    if frac:
        if not frac.isdigit():
            raise ValueError(f"Unrecognised fractional seconds in timestamp: {s!r}")
        ns += int((frac + "000000000")[:9])
    offset_ns = _zone_offset_ns(zone)
    return ns - offset_ns, offset_ns


def _format_ns(ns: int) -> str:
    """Epoch ns -> the whole-seconds prefix, rendered in UTC."""
    return datetime.fromtimestamp(ns // _NS_PER_SEC, tz=timezone.utc).strftime(_TS_FMT)


# ---------------------------------------------------------------------------
# Windowed streaming read
# ---------------------------------------------------------------------------

def _read_window(src: Path, offset_hours: float, num_reqs: int) -> dict:
    """Stream ``src`` and return the first ``num_reqs`` rows at or after
    ``first_timestamp + offset_hours``.

    The whole-seconds prefix of the timestamp is fixed-width and
    zero-padded, so rows before the window are skipped with a string
    comparison and only candidates near the boundary are parsed.
    """
    rows: list[tuple[int, int, int]] = []
    first_ns: int | None = None
    zone_offset_ns = 0
    ts_len = 0
    target_ns: int | None = None
    target_prefix = ""
    window_line_no: int | None = None
    max_prefix = ""
    out_of_order_prefix = 0
    ts_shape_drift = 0

    with src.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().strip().lstrip("﻿")
        fields = tuple(c.strip() for c in header.split(","))
        if fields != EXPECTED_HEADER:
            raise ValueError(
                f"Unexpected header in {src.name}: {fields}. "
                f"Expected {EXPECTED_HEADER} (2024 Azure LLM inference schema).")

        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            ts_text, sep, rest = line.partition(",")
            if not sep:
                raise ValueError(f"{src.name}:{line_no}: malformed row: {line!r}")

            if len(ts_text) < _TS_PREFIX_LEN:
                raise ValueError(
                    f"{src.name}:{line_no}: timestamp too short: {ts_text!r}")
            prefix = ts_text[:_TS_PREFIX_LEN]
            if prefix < max_prefix:
                out_of_order_prefix += 1
            else:
                max_prefix = prefix

            if first_ns is None:
                first_ns, zone_offset_ns = _parse_timestamp_ns(ts_text)
                ts_len = len(ts_text)
                target_ns = first_ns + round(offset_hours * _NS_PER_HOUR)
                # The prefix gate compares the text as written, so the target
                # is rendered in the file's own zone, not in UTC.
                target_prefix = _format_ns(target_ns + zone_offset_ns)
            elif len(ts_text) != ts_len:
                # Cheap shape check: the gate below assumes every row is
                # written the same way. Reported, and enforced exactly on
                # every row that is actually parsed.
                ts_shape_drift += 1

            # Cheap gate: skip everything whose whole second is before the
            # window. Only boundary rows reach strptime.
            if not rows and prefix < target_prefix:
                continue

            ts_ns, row_zone_ns = _parse_timestamp_ns(ts_text)
            if row_zone_ns != zone_offset_ns:
                raise ValueError(
                    f"{src.name}:{line_no}: timestamp zone offset changed "
                    f"mid-file ({ts_text!r}). The window gate compares "
                    f"timestamps as text, which is only valid under one zone.")
            if not rows:
                if ts_ns < target_ns:
                    continue
                window_line_no = line_no

            ctx_text, sep2, gen_text = rest.partition(",")
            if not sep2:
                raise ValueError(f"{src.name}:{line_no}: malformed row: {line!r}")
            rows.append((ts_ns, int(ctx_text), int(gen_text.split(",")[0])))
            if len(rows) >= num_reqs:
                break

    if first_ns is None:
        raise ValueError(f"{src.name} has a header but no data rows.")
    if not rows:
        raise ValueError(
            f"{src.name}: no row at or after {_format_ns(target_ns)} "
            f"(first timestamp {_format_ns(first_ns)} + {offset_hours:g} h). "
            f"The window offset exceeds the trace.")
    if len(rows) < num_reqs:
        raise ValueError(
            f"{src.name}: window starting {_format_ns(target_ns)} yields only "
            f"{len(rows)} rows, fewer than the requested {num_reqs}. §5.2 "
            f"requires one contiguous window; refusing to wrap or pad.")

    return {
        "rows": rows,
        "first_ns": first_ns,
        "target_ns": target_ns,
        "window_line_no": window_line_no,
        "out_of_order_prefix": out_of_order_prefix,
        "ts_shape_drift": ts_shape_drift,
        "zone_offset_ns": zone_offset_ns,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _median(values: list[int]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _summarize(window: dict, arrivals: list[int], scale: float) -> dict:
    rows = window["rows"]
    ts = [r[0] for r in rows]
    inp = [r[1] for r in rows]
    out = [r[2] for r in rows]

    native_span_ns = ts[-1] - ts[0]
    scaled_span_ns = arrivals[-1] - arrivals[0]
    n = len(rows)

    def _rate(span_ns: int) -> float:
        # Mean arrival rate over the window: n-1 gaps across the span.
        return (n - 1) / (span_ns / _NS_PER_SEC) if span_ns > 0 else float("inf")

    non_monotonic = sum(1 for a, b in zip(arrivals, arrivals[1:]) if b < a)

    return {
        "num_reqs": n,
        "window_first_line": window["window_line_no"],
        "window_start": _format_ns(ts[0]),
        "window_end": _format_ns(ts[-1]),
        "native_span_sec": native_span_ns / _NS_PER_SEC,
        "scaled_span_sec": scaled_span_ns / _NS_PER_SEC,
        "native_rate_rps": _rate(native_span_ns),
        "scaled_rate_rps": _rate(scaled_span_ns),
        "input_toks_mean": sum(inp) / n,
        "input_toks_median": _median(inp),
        "input_toks_min": min(inp),
        "input_toks_max": max(inp),
        "output_toks_mean": sum(out) / n,
        "output_toks_median": _median(out),
        "output_toks_min": min(out),
        "output_toks_max": max(out),
        "rows_with_zero_input_toks": sum(1 for v in inp if v <= 0),
        "rows_with_zero_output_toks": sum(1 for v in out if v <= 0),
        "non_monotonic_arrivals": non_monotonic,
        "out_of_order_rows_scanned": window["out_of_order_prefix"],
        "timestamp_shape_drift_rows": window["ts_shape_drift"],
        "timestamp_zone_offset_sec": window["zone_offset_ns"] // _NS_PER_SEC,
    }


def _report(src: Path, out_path: Path, args: argparse.Namespace,
            scale: float, s: dict) -> None:
    print(f"Source            {src}")
    print(f"Window            {s['window_start']} .. {s['window_end']} "
          f"(first timestamp + {args.window_offset_hours:g} h)")
    print(f"Requests          {s['num_reqs']}")
    print(f"Native  span/rate {s['native_span_sec']:.1f} s / "
          f"{s['native_rate_rps']:.3f} req/s")
    print(f"Scaled  span/rate {s['scaled_span_sec']:.1f} s / "
          f"{s['scaled_rate_rps']:.3f} req/s   (scale {args.scale})")
    print(f"input_toks        mean {s['input_toks_mean']:.1f}  "
          f"median {s['input_toks_median']:.1f}  "
          f"min {s['input_toks_min']}  max {s['input_toks_max']}")
    print(f"output_toks       mean {s['output_toks_mean']:.1f}  "
          f"median {s['output_toks_median']:.1f}  "
          f"min {s['output_toks_min']}  max {s['output_toks_max']}")

    # Reported, never acted on: §5.2 forbids filtering and truncation.
    if s["rows_with_zero_input_toks"]:
        print(f"NOTE  {s['rows_with_zero_input_toks']} row(s) have "
              f"input_toks <= 0, emitted unchanged (no filtering per §5.2).")
    if s["rows_with_zero_output_toks"]:
        print(f"NOTE  {s['rows_with_zero_output_toks']} row(s) have "
              f"output_toks <= 0, emitted unchanged (no filtering per §5.2).")
    if s["non_monotonic_arrivals"]:
        print(f"NOTE  {s['non_monotonic_arrivals']} arrival(s) go backwards "
              f"within the window; emitted in file order, the router sorts.")
    if s["out_of_order_rows_scanned"]:
        print(f"NOTE  {s['out_of_order_rows_scanned']} row(s) scanned before the "
              f"window were out of timestamp order.")
    if s["timestamp_shape_drift_rows"]:
        print(f"NOTE  {s['timestamp_shape_drift_rows']} row(s) scanned have a "
              f"differently-shaped timestamp than the first row.")


def _manifest(src: Path, out_path: Path, args: argparse.Namespace,
              scale: float, s: dict) -> dict:
    st = src.stat()
    return {
        "generator": "azure",
        "preregistration": (
            "thesis-agentic-sharding/fleetsieve-moving-oracle-audit.md "
            "§5.2 (ingestion), §5.3 (rate ladder)"),
        "source": {
            "path": str(src),
            "name": src.name,
            "size_bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "schema": list(EXPECTED_HEADER),
        },
        "frozen": {
            "window_offset_hours": WINDOW_OFFSET_HOURS,
            "num_reqs": NUM_REQS,
            "scale_ladder": [f"1/{round(1 / v)}" if v != 1.0 else "1"
                             for v in SCALE_LADDER],
        },
        "used": {
            "window_offset_hours": args.window_offset_hours,
            "num_reqs": args.num_reqs,
            "scale": scale,
            "scale_arg": args.scale,
            "deviates_from_frozen": bool(
                args.window_offset_hours != WINDOW_OFFSET_HOURS
                or args.num_reqs != NUM_REQS),
        },
        "output": str(out_path),
        "summary": s,
    }
