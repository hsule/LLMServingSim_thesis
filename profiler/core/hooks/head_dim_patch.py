"""Pin ``head_dim`` for architectures that derive it from hidden_size.

Background
----------
The profiler emulates TP=N on a single GPU by dividing ``SHARD_FIELDS``
(num_attention_heads, num_key_value_heads, intermediate_size, vocab_size)
and booting vLLM with ``tensor_parallel_size=1``. ``hidden_size`` is
deliberately left alone: a real TP rank still sees the full hidden state
on the wire, and shrinking it would corrupt qkv_proj's input width,
o_proj's output width, the layernorms and the MoE block.

Most architectures read an explicit ``head_dim`` from the HF config when
one is present::

    head_dim = getattr(config, "head_dim", None)
    self.head_dim = head_dim or self.hidden_size // self.total_num_heads

For those, ``engine.fuse_engine_kwargs`` pinning ``head_dim`` into
``hf_overrides`` is enough. Three families in vLLM 0.19.0 instead derive
it unconditionally and ignore the config field entirely::

    self.head_dim = self.hidden_size // self.total_num_heads

    vllm/model_executor/models/olmoe.py:149      OlmoeAttention
    vllm/model_executor/models/qwen2_moe.py:226  Qwen2MoeAttention
    vllm/model_executor/models/granitemoe.py:166 GraniteMoeAttention

For those, dividing num_attention_heads makes head_dim *grow* by the same
factor, so per-rank attention gets wider instead of narrower. A 16-head /
2048-hidden model at tp=8 emulates 2 heads x 1024 = the full 2048-wide q
of tp=1 -- TP is not emulated at all -- and FlashAttention rejects any
head size above 256, so tp=4 and tp=8 fail outright.

The fix
-------
Install a data descriptor for ``head_dim`` on the affected attention
classes. ``self.head_dim = <derived>`` inside ``__init__`` is then
swallowed and every later read returns the pinned value, so the whole
downstream chain lands on real per-rank geometry::

    self.q_size   = self.num_heads * self.head_dim
    self.kv_size  = self.num_kv_heads * self.head_dim
    self.scaling  = self.head_dim ** -0.5
    QKVParallelLinear(self.hidden_size, self.head_dim, total_heads, total_kv)
    RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size)
    get_rope(self.head_dim, rotary_dim=self.head_dim, ...)
    Attention(self.num_heads, self.head_dim, self.scaling, ...)

A data descriptor is used rather than a vendored copy of ``__init__``
because it depends only on *that one assignment existing*, not on the
surrounding source text, so it survives vLLM point releases that the
existing ``FusedMoE.forward_native`` patch would not.

At tp=1 the pinned value equals the derived value, so installing the
descriptor is a numerical no-op and tp=1 profiles stay bit-identical.

The pinned value crosses the host/worker process boundary through
``PROFILER_PINNED_HEAD_DIM`` in the environment, because vLLM may run
the model in a spawned EngineCore process that does not inherit
host-side monkeypatches (subprocesses do inherit ``os.environ``).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

#: Environment variable carrying the pinned head_dim from host to worker.
ENV_PINNED_HEAD_DIM = "PROFILER_PINNED_HEAD_DIM"

#: (module path, attention class name) for architectures whose vLLM
#: implementation derives head_dim and ignores ``config.head_dim``.
#: Families that honour the config field are deliberately absent -- they
#: are already correct via hf_overrides and must not be touched.
DERIVED_HEAD_DIM_CLASSES: tuple[tuple[str, str], ...] = (
    ("vllm.model_executor.models.olmoe", "OlmoeAttention"),
    ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeAttention"),
    ("vllm.model_executor.models.granitemoe", "GraniteMoeAttention"),
)


class PinnedHeadDim:
    """Data descriptor that reports a fixed ``head_dim``.

    Assignment is swallowed but remembered under ``_derived_head_dim``
    so the discarded value stays visible for debugging. Being a *data*
    descriptor (it defines ``__set__``) it takes precedence over the
    instance ``__dict__``, so the assignment inside ``__init__`` cannot
    shadow it.
    """

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return self.value

    def __set__(self, obj, value) -> None:
        try:
            obj.__dict__["_derived_head_dim"] = value
        except AttributeError:  # pragma: no cover - slotted modules
            pass


def apply_pinned_head_dim(head_dim: int) -> list[str]:
    """Install the descriptor on every derived-head_dim attention class.

    Classes whose module is not importable in this vLLM build are
    skipped silently -- the catalogs we ship do not require all of them.

    Returns the list of ``module.Class`` names actually patched.
    """
    import importlib

    patched: list[str] = []
    for module_path, class_name in DERIVED_HEAD_DIM_CLASSES:
        try:
            module = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001 - arch absent from this build
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        setattr(cls, "head_dim", PinnedHeadDim(head_dim))
        patched.append(f"{module_path.rsplit('.', 1)[-1]}.{class_name}")

    if patched:
        log.debug("pinned head_dim=%d on %s", head_dim, ", ".join(patched))
    return patched


def maybe_apply_from_env() -> list[str]:
    """Apply the pin if the host set ``PROFILER_PINNED_HEAD_DIM``.

    Called at worker-extension import time, which vLLM resolves before
    the model is constructed.
    """
    raw = os.environ.get(ENV_PINNED_HEAD_DIM)
    if not raw:
        return []
    try:
        head_dim = int(raw)
    except ValueError:
        log.warning("%s=%r is not an int; ignoring", ENV_PINNED_HEAD_DIM, raw)
        return []
    if head_dim <= 0:
        return []
    return apply_pinned_head_dim(head_dim)
