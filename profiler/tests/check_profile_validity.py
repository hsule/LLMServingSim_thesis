"""Check a profiled bundle for the invariants TP emulation must satisfy.

Run from the repo root::

    python3 -m profiler.tests.check_profile_validity \
        profiler/perf/RTX4090/allenai/OLMoE-1B-7B-0125-Instruct/bf16

This is the empirical counterpart to ``test_tp_geometry.py``. That test
checks the *intended* geometry from the config; this one checks the
*measured* CSVs actually behave like per-rank shards.

The head_dim bug is directly visible here. When head_dim grew with TP,
per-rank q width stayed at the tp=1 value, so attention and the qkv/o
projections cost the *same* at every TP degree. A correct emulation
makes per-rank work shrink roughly as 1/TP.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

# Layers whose per-rank cost should fall as TP rises (column-/row-parallel).
TP_SCALING_LAYERS = ("qkv_proj", "o_proj")
# Layers profiled once at tp=1 and replicated; must be identical across TPs.
EXPECT_REPLICATED = ("layernorm", "final_layernorm", "qk_norm", "sampler")

FAILURES: list[str] = []
NOTES: list[str] = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL {msg}")


def note(msg):
    NOTES.append(msg)
    print(f"  note {msg}")


def read_rows(path):
    if not path.is_file():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def dense_by_layer(root, tp):
    rows = read_rows(root / f"tp{tp}" / "dense.csv")
    out = {}
    for r in rows:
        out.setdefault(r["layer"], {})[int(r["tokens"])] = float(r["time_us"])
    return out


def attention_total(root, tp):
    """Sum of profiled attention times over the shared key set."""
    rows = read_rows(root / f"tp{tp}" / "attention.csv")
    return {
        (r["prefill_chunk"], r["kv_prefill"], r["n_decode"], r["kv_decode"]):
            float(r["time_us"]) for r in rows
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"no such bundle: {root}")
        return 2

    meta = yaml.safe_load((root / "meta.yaml").read_text())
    tps = sorted(
        int(p.name[2:]) for p in root.iterdir()
        if p.is_dir() and p.name.startswith("tp") and p.name[2:].isdigit()
    )
    print(f"bundle : {root}")
    print(f"gpu    : {meta.get('gpu')}   hardware: {meta.get('hardware')}")
    print(f"vllm   : {meta.get('vllm_version')}")
    print(f"tps    : {tps}")

    ov = (meta.get("engine_effective") or {}).get("hf_overrides", {})
    print(f"\n=== 1. head_dim pinned in overrides ===")
    hd = ov.get("head_dim")
    if hd is None:
        note("meta.yaml records no head_dim override. Correct only if the "
             "model config declares head_dim itself (e.g. Qwen3).")
    else:
        print(f"  head_dim = {hd}")
        if hd % 8 or hd > 256:
            fail(f"head_dim={hd} violates FlashAttention "
                 f"(needs %8==0 and <=256)")

    # ---- 2. per-rank work must shrink with TP ----
    print(f"\n=== 2. per-rank cost scales down with TP ===")
    base = dense_by_layer(root, tps[0])
    print(f"{'layer':16} " + " ".join(f"{'tp%d'%t:>12}" for t in tps)
          + f"   {'tp%d/tp%d' % (tps[-1], tps[0]):>10}")
    for layer in TP_SCALING_LAYERS:
        if layer not in base:
            continue
        # compare at the largest token count present in every tp
        common = set(base[layer])
        per_tp = {}
        for t in tps:
            d = dense_by_layer(root, t).get(layer, {})
            per_tp[t] = d
            common &= set(d)
        if not common:
            note(f"{layer}: no shared token count across TPs; skipped")
            continue
        tok = max(common)
        vals = [per_tp[t][tok] for t in tps]
        ratio = vals[-1] / vals[0] if vals[0] else float("nan")
        print(f"{layer:16} " + " ".join(f"{v:12.2f}" for v in vals)
              + f"   {ratio:10.3f}   (tokens={tok})")
        if len(tps) > 1:
            # A correct shard makes the widest TP cheaper. The bug made
            # this ratio ~1.0 because per-rank width never changed.
            if ratio > 0.95:
                fail(f"{layer}: tp{tps[-1]}/tp{tps[0]} = {ratio:.3f}; per-rank "
                     f"cost did not fall with TP — geometry is probably not "
                     f"being sharded (the head_dim bug looks like this)")
            elif ratio > 0.75:
                note(f"{layer}: ratio {ratio:.3f} is higher than the ~1/TP "
                     f"ideal; small layers are launch-latency bound, so this "
                     f"is only suspicious if it is near 1.0")

    # ---- 3. attention must shrink with TP ----
    print(f"\n=== 3. attention cost scales down with TP ===")
    a0 = attention_total(root, tps[0])
    print(f"{'tp':>4} {'shots':>7} {'mean us (shared keys)':>22} {'vs tp%d'%tps[0]:>10}")
    shared = set(a0)
    per_tp_attn = {}
    for t in tps:
        at = attention_total(root, t)
        per_tp_attn[t] = at
        shared &= set(at)
    if not shared:
        note("no attention keys shared across TP folders; skipped")
    else:
        base_mean = sum(a0[k] for k in shared) / len(shared)
        for t in tps:
            m = sum(per_tp_attn[t][k] for k in shared) / len(shared)
            r = m / base_mean if base_mean else float("nan")
            print(f"{t:>4} {len(per_tp_attn[t]):>7} {m:22.3f} {r:10.3f}")
            if t == tps[-1] and len(tps) > 1 and r > 0.95:
                fail(f"attention tp{t}/tp{tps[0]} = {r:.3f}; per-rank attention "
                     f"did not shrink — head_dim is probably not pinned")

    # ---- 4. tp_stable layers must be identical everywhere ----
    print(f"\n=== 4. tp_stable layers replicated identically ===")
    for layer in EXPECT_REPLICATED:
        if layer not in base:
            continue
        vals = []
        ok = True
        for t in tps:
            d = dense_by_layer(root, t).get(layer, {})
            if not d:
                d = {int(r["sequences"]): float(r["time_us"])
                     for r in read_rows(root / f"tp{t}" / "per_sequence.csv")
                     if r["layer"] == layer}
            vals.append(d)
        common = set.intersection(*[set(v) for v in vals]) if vals else set()
        if not common:
            continue
        k = max(common)
        series = [v[k] for v in vals]
        if len(set(series)) != 1:
            ok = False
        print(f"  {layer:16} " + " ".join(f"{s:10.3f}" for s in series)
              + ("   identical" if ok else "   DIFFERS"))
        if not ok:
            fail(f"{layer} is declared tp_stable but differs across TP folders")

    # ---- 5. required files ----
    print(f"\n=== 5. required files ===")
    for t in tps:
        d = root / f"tp{t}"
        have = sorted(p.name for p in d.glob("*.csv"))
        print(f"  tp{t}: {', '.join(have)}")
        for req in ("dense.csv", "per_sequence.csv", "attention.csv"):
            if req not in have:
                fail(f"tp{t} missing {req}")
    moe1 = root / f"tp{tps[0]}" / "moe.csv"
    if moe1.is_file():
        print(f"  moe.csv present at tp{tps[0]} "
              f"(the simulator reads MoE only from tp1)")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"PROFILE INVALID: {len(FAILURES)} failed check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PROFILE VALID" + (f"  ({len(NOTES)} note(s))" if NOTES else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
