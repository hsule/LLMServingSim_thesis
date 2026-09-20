#!/usr/bin/env python3
"""ZIPF expert-popularity routing: a controlled compute-imbalance knob.

Scope note, deliberately repeated here and in the policy's docstring: the
collective sizes in ``_emit_moe_block`` are computed from ``total_len`` and
``ep_total`` and never read the routing result. ZIPF therefore imbalances each
EP rank's ``local_tokens`` -- and so the slowest-rank critical path -- and
nothing else. It is not a model of real MoE routing skew.
"""
import os
import statistics as st
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from serving.core.gate_function import GateRouter  # noqa: E402

E, K = 64, 8            # OLMoE
TOKENS = 2048           # max_num_batched_tokens


def router(alpha, seed=42, block_copy=True):
    return GateRouter(0, 0, E, num_experts_per_tok=K,
                      routing_policy=f"ZIPF:{alpha}", seed=seed, block_copy=block_copy)


def cov(xs):
    m = st.mean(xs)
    return (st.pstdev(xs) / m) if m else 0.0


class TestParsing(unittest.TestCase):
    def test_alpha_rides_on_the_policy_string(self):
        r = router(1.0)
        self.assertEqual(r.routing_policy, "ZIPF")
        self.assertAlmostEqual(r.skew_alpha, 1.0)

    def test_bare_policies_still_parse(self):
        for p in ("BALANCED", "RR", "RAND"):
            g = GateRouter(0, 0, E, num_experts_per_tok=K, routing_policy=p)
            self.assertEqual(g.routing_policy, p)
            self.assertEqual(g.skew_alpha, 0.0)

    def test_negative_alpha_raises(self):
        with self.assertRaises(ValueError):
            router(-0.5)


class TestAlphaZeroIsUniform(unittest.TestCase):
    def test_expert_frequencies_are_approximately_uniform(self):
        # Not bit-identical to BALANCED: alpha=0 is uniform *random*, so this
        # asserts statistical uniformity over a large deterministic sample.
        r = router(0.0)
        counts = [0] * E
        for t in range(8000):
            for e in r.routing_fn(t, E, K):
                counts[e] += 1
        expected = 8000 * K / E
        self.assertLess(cov(counts), 0.05,
                        f"alpha=0 should be near-uniform, CoV={cov(counts):.4f}")
        self.assertLess(abs(max(counts) - expected) / expected, 0.15)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_routing(self):
        a = [router(1.0, seed=7).routing_fn(t, E, K) for t in range(50)]
        b = [router(1.0, seed=7).routing_fn(t, E, K) for t in range(50)]
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = [router(1.0, seed=7).routing_fn(t, E, K) for t in range(50)]
        c = [router(1.0, seed=8).routing_fn(t, E, K) for t in range(50)]
        self.assertNotEqual(a, c)

    def test_popularity_is_fixed_across_calls(self):
        # The same experts stay hot: weights are drawn once in __init__.
        r = router(1.0)
        self.assertEqual(r._zipf_weights, router(1.0)._zipf_weights)


class TestImbalanceMonotonicity(unittest.TestCase):
    ALPHAS = (0.0, 0.5, 1.0)

    def _ranks(self, alpha, ep):
        res = router(alpha).route_ep(0, "b", TOKENS, ep)
        return res.local_tokens

    def test_imbalance_rises_with_alpha_at_ep8(self):
        covs = [cov(self._ranks(a, 8)) for a in self.ALPHAS]
        for lo, hi in zip(covs, covs[1:]):
            self.assertLessEqual(lo, hi + 1e-9,
                                 f"imbalance must not fall with alpha: {covs}")
        self.assertGreater(covs[-1], covs[0],
                           f"alpha=1.0 must be more imbalanced than alpha=0: {covs}")

    def test_ep1_has_no_cross_rank_imbalance(self):
        for a in self.ALPHAS:
            self.assertEqual(len(self._ranks(a, 1)), 1)
            self.assertEqual(cov(self._ranks(a, 1)), 0.0)


class TestConservation(unittest.TestCase):
    def test_expert_token_assignments_are_constant(self):
        # Every token picks exactly k distinct experts at every alpha.
        for a in (0.0, 0.5, 1.0):
            r = router(a)
            for t in range(200):
                sel = r.routing_fn(t, E, K)
                self.assertEqual(len(sel), K)
                self.assertEqual(len(set(sel)), K)

    def test_source_tokens_are_unchanged_by_alpha(self):
        for ep in (1, 2, 4, 8):
            base = router(0.0).route_ep(0, "b", TOKENS, ep).source_tokens
            for a in (0.5, 1.0):
                self.assertEqual(router(a).route_ep(0, "b", TOKENS, ep).source_tokens,
                                 base, f"ep={ep} alpha={a}")


class TestProfileRange(unittest.TestCase):
    """No lookup may fall outside the profiled MoE grid.

    OLMoE tp1/moe.csv is tokens in [1, 2048] x activated_experts in
    {8, 16, 32, 64}. Below 8 the lookup silently clamps, so a rank activating
    fewer than 8 distinct experts would be modelled as if it activated 8.
    """
    TOK_MAX = 2048
    AE_MIN = 8

    def test_queries_stay_inside_the_grid(self):
        for a in (0.0, 0.5, 1.0):
            for ep in (1, 2, 4, 8):
                res = router(a).route_ep(0, "b", self.TOK_MAX, ep)
                for rank, (lt, ae) in enumerate(
                        zip(res.local_tokens, res.activated_experts)):
                    self.assertLessEqual(lt, self.TOK_MAX,
                                         f"alpha={a} ep={ep} rank={rank} tokens={lt}")
                    self.assertGreaterEqual(
                        ae, self.AE_MIN,
                        f"alpha={a} ep={ep} rank={rank} activated={ae} is below the "
                        f"profiled minimum {self.AE_MIN}; the lookup would clamp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
