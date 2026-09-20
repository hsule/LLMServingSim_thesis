"""Self-tests for TP-emulation attention geometry.

Run from the repo root inside the vLLM container (config.py needs
pydantic + pyyaml)::

    python3 -m profiler.tests.test_tp_geometry

The profiler emulates TP=N on one GPU by dividing SHARD_FIELDS and
booting vLLM at ``tensor_parallel_size=1``. These tests lock in the
invariant that makes that emulation faithful:

    1. real head_dim is CONSTANT across TP degrees
    2. heads per rank divides by TP
    3. q/k/v projection width per rank divides by TP
    4. TP1 geometry is identical to the unsharded config
    5. head_dim stays inside FlashAttention's supported range

Covered config styles, which differ in how vLLM recovers head_dim:

    qwen3   explicit ``head_dim`` in the HF config; vLLM honours it.
            Note 2048/32 = 64 != 128, so the explicit value is load
            bearing and must never be overwritten by the pin.
    llama   no ``head_dim`` in the JSON, but transformers'
            ``LlamaConfig.__init__`` MATERIALIZES the attribute
            (``hidden_size // num_attention_heads``) at construction,
            before vLLM applies hf_overrides -- and vLLM reads it via
            ``getattr(config, "head_dim", None) or ...``. So Llama was
            already correct; the injected pin is the same value and a
            no-op. Kept in the matrix as the regression guard for that.
    olmoe   no ``head_dim`` in the JSON, ``OlmoeConfig`` does not
            materialize one (it is None), and ``olmoe.py`` computes
            ``hidden // total_num_heads`` unconditionally, never
            consulting the config. This is the family that was actually
            broken, and the pin only lands via hooks.head_dim_patch's
            data descriptor.

Measured with transformers 4.57.6 / vLLM 0.19.0:

    Llama-3.1-8B  json head_dim absent, config object head_dim = 128
    OLMoE-1B-7B   json head_dim absent, config object head_dim = None
"""

from __future__ import annotations

import sys

from profiler.core.config import compute_sharded_overrides
from profiler.core.hooks.head_dim_patch import PinnedHeadDim

# FlashAttentionBackend.supports_head_size() in vLLM 0.19.0
FA_MAX_HEAD_SIZE = 256


def fa_supports(head_size: int) -> bool:
    return head_size % 8 == 0 and head_size <= FA_MAX_HEAD_SIZE


# --------------------------------------------------------------------------
# Model configs under test (real checkpoints)
# --------------------------------------------------------------------------

QWEN3_MOE = {
    "name": "Qwen3-30B-A3B-Instruct-2507",
    "style": "honors",
    "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
    "intermediate_size": 6144, "vocab_size": 151936, "head_dim": 128,
}
LLAMA = {
    "name": "Llama-3.1-8B",
    "style": "honors",
    "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 14336, "vocab_size": 128256,
}
OLMOE = {
    "name": "OLMoE-1B-7B-0125-Instruct",
    "style": "derives",
    "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 16,
    "intermediate_size": 1024, "vocab_size": 50304,
}


def model_cfg(spec):
    """The HF-config subset, without our test-only bookkeeping keys."""
    return {k: v for k, v in spec.items() if k not in ("name", "style")}


def effective_head_dim(spec, overrides):
    """head_dim vLLM ends up using, per architecture style.

    ``honors``  -> getattr(config, "head_dim", None) or hidden // heads
    ``derives`` -> hidden // total_num_heads, config field ignored;
                   hooks.head_dim_patch then pins it to the value the
                   host exported, which is the same one we injected.
    """
    merged = {**model_cfg(spec), **overrides}
    if spec["style"] == "honors":
        return merged.get("head_dim") or (
            merged["hidden_size"] // merged["num_attention_heads"]
        )
    pinned = overrides.get("head_dim") or model_cfg(spec).get("head_dim")
    if pinned:  # hooks.head_dim_patch installed the descriptor
        return pinned
    return merged["hidden_size"] // merged["num_attention_heads"]


def geometry(spec, tp):
    """Per-rank attention geometry the emulated engine will build."""
    ov = compute_sharded_overrides(model_cfg(spec), tp)
    hd = effective_head_dim(spec, ov)
    heads = ov["num_attention_heads"]
    kv = ov["num_key_value_heads"]
    return {"head_dim": hd, "heads": heads, "kv_heads": kv,
            "q_size": heads * hd, "kv_size": kv * hd, "overrides": ov}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

FAILURES: list[str] = []


def check(cond, msg):
    if cond:
        return True
    FAILURES.append(msg)
    print(f"  FAIL {msg}")
    return False


def test_invariants(spec, tps=(1, 2, 4, 8)):
    name = spec["name"]
    print(f"\n=== {name}  ({spec['style']} head_dim) ===")
    base = geometry(spec, 1)
    base_hd = model_cfg(spec).get("head_dim") or (
        spec["hidden_size"] // spec["num_attention_heads"]
    )

    # (4) TP1 must reproduce the unsharded config exactly.
    check(base["head_dim"] == base_hd,
          f"{name} tp1 head_dim {base['head_dim']} != unsharded {base_hd}")
    check(base["heads"] == spec["num_attention_heads"],
          f"{name} tp1 heads changed")
    check(base["kv_heads"] == spec["num_key_value_heads"],
          f"{name} tp1 kv_heads changed")

    print(f"{'tp':>3} {'heads':>6} {'kv':>4} {'head_dim':>9} {'q_size':>7} "
          f"{'kv_size':>8}  {'FA ok':>6}")
    for tp in tps:
        try:
            g = geometry(spec, tp)
        except ValueError as e:
            print(f"{tp:>3}  rejected: {e}")
            # Rejection is correct behaviour when a field cannot divide;
            # Qwen3-30B-A3B has kv_heads=4 and genuinely cannot do tp8.
            check(tp == 8 and spec is QWEN3_MOE,
                  f"{name} tp{tp} unexpectedly rejected: {e}")
            continue
        ok_fa = fa_supports(g["head_dim"])
        print(f"{tp:>3} {g['heads']:>6} {g['kv_heads']:>4} {g['head_dim']:>9} "
              f"{g['q_size']:>7} {g['kv_size']:>8}  {str(ok_fa):>6}")

        # (1) head_dim constant across TP
        check(g["head_dim"] == base_hd,
              f"{name} tp{tp}: head_dim {g['head_dim']} != {base_hd} "
              f"(must be TP-invariant)")
        # (2) heads per rank divides by TP
        check(g["heads"] == spec["num_attention_heads"] // tp,
              f"{name} tp{tp}: heads {g['heads']} != "
              f"{spec['num_attention_heads'] // tp}")
        # (3) q/kv projection width per rank divides by TP
        check(g["q_size"] == base["q_size"] // tp,
              f"{name} tp{tp}: q_size {g['q_size']} != {base['q_size'] // tp}")
        check(g["kv_size"] == base["kv_size"] // tp,
              f"{name} tp{tp}: kv_size {g['kv_size']} != "
              f"{base['kv_size'] // tp}")
        # (5) FlashAttention must accept the head size
        check(ok_fa, f"{name} tp{tp}: head_dim {g['head_dim']} unsupported "
                     f"by FlashAttention (needs %8==0 and <=256)")


def test_explicit_head_dim_preserved():
    """A config that declares head_dim is authoritative; never overwrite."""
    print("\n=== explicit head_dim is preserved ===")
    for tp in (1, 2, 4):
        ov = compute_sharded_overrides(model_cfg(QWEN3_MOE), tp)
        check("head_dim" not in ov,
              f"pin must not be injected when the config declares head_dim "
              f"(tp={tp}, got {ov.get('head_dim')})")
    derived = QWEN3_MOE["hidden_size"] // QWEN3_MOE["num_attention_heads"]
    check(derived != QWEN3_MOE["head_dim"],
          "Qwen3 fixture should have head_dim != hidden/heads to be meaningful")
    print(f"  ok: head_dim={QWEN3_MOE['head_dim']} kept, "
          f"derived would have been {derived}")


# What each architecture's head_dim resolved to BEFORE the pin existed.
# Two independent things have to fail for a family to have been broken:
# the transformers config must not materialize head_dim, AND the vLLM
# model class must not consult it. Only the second column's "derives"
# families satisfy both.
PRE_FIX_HEAD_DIM = {
    # name: (config object materialized head_dim?, vLLM consults it?)
    "Qwen3-30B-A3B-Instruct-2507": (True, True),    # explicit in the JSON
    "Llama-3.1-8B": (True, True),                   # LlamaConfig computes it
    "OLMoE-1B-7B-0125-Instruct": (False, False),    # neither
}


def test_regression_which_families_were_broken():
    """Pin down which families the pre-fix code actually got wrong.

    Getting this wrong in either direction is expensive: it either
    condemns good committed profiles or lets bad ones stand.
    """
    print("\n=== regression: which families were broken pre-fix ===")
    print(f"{'model':28} {'cfg has hd':>11} {'vllm reads':>11} {'pre-fix tp8':>12} {'broken':>8}")
    for spec in (QWEN3_MOE, LLAMA, OLMOE):
        name = spec["name"]
        materialized, consults = PRE_FIX_HEAD_DIM[name]
        base_hd = model_cfg(spec).get("head_dim") or (
            spec["hidden_size"] // spec["num_attention_heads"]
        )
        # Pre-fix, head_dim came from the config when both held, else it
        # was derived from the *sharded* head count.
        if materialized and consults:
            old_hd = base_hd
        else:
            heads8 = spec["num_attention_heads"] // 8
            old_hd = spec["hidden_size"] // heads8 if heads8 else None
        broken = old_hd != base_hd
        print(f"{name:28} {str(materialized):>11} {str(consults):>11} "
              f"{str(old_hd):>12} {str(broken):>8}")
        check(broken == (name == OLMOE["name"]),
              f"{name}: expected broken={name == OLMOE['name']}, got {broken}")
    print("  -> only the olmoe-style families (olmoe / qwen2_moe / granitemoe,")
    print("     which ignore config.head_dim entirely) were actually wrong.")
    print("     Committed Llama and Qwen3 tp2 profiles are NOT affected.")


def test_descriptor_pins():
    """The data descriptor swallows the derived assignment."""
    print("\n=== hooks.head_dim_patch descriptor ===")

    class FakeAttention:
        pass

    FakeAttention.head_dim = PinnedHeadDim(128)
    a = FakeAttention()
    a.head_dim = 1024                     # what OlmoeAttention.__init__ does
    check(a.head_dim == 128,
          f"descriptor did not pin: got {a.head_dim}")
    check(a.__dict__.get("_derived_head_dim") == 1024,
          "descriptor should record the discarded derived value")
    # downstream chain now lands on real per-rank geometry
    num_heads = 2
    check(num_heads * a.head_dim == 256,
          "q_size via pinned head_dim should be 256")
    print("  ok: assignment of 1024 swallowed, head_dim=128, q_size=256")


def main():
    test_invariants(QWEN3_MOE)
    test_invariants(LLAMA)
    test_invariants(OLMOE)
    test_explicit_head_dim_preserved()
    test_regression_which_families_were_broken()
    test_descriptor_pins()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL TP GEOMETRY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
