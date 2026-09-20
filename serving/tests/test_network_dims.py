#!/usr/bin/env python3
"""Physical-topology (`network_dims`) and derived involved_dim invariants.

The declared topology is what lets a TP/EP group either sit inside a physical
node or straddle its boundary while every other input stays identical. These
tests pin the two things that make that trustworthy: the address decomposition
must match ASTRA-Sim's, and a group that cannot be written as a product of
topology dimensions must raise instead of being approximated.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from serving.core.config_builder import (  # noqa: E402
    _compute_network_dims, _decompose, _dims_spanned, _validate_network_dims,
)


def inst(num_npus, **kw):
    d = {"num_npus": num_npus, "tp_size": num_npus, "pp_size": 1,
         "ep_size": num_npus, "pd_type": None}
    d.update(kw)
    return d


class TestDecompose(unittest.TestCase):
    def test_matches_astra_sim_rule(self):
        # MultiDimTopology::translate_address, innermost dimension first.
        dims = [2, 8, 4]
        self.assertEqual(_decompose(47, dims), [1, 7, 2])

    def test_round_trip_over_a_2x2x2_cube(self):
        dims = [2, 2, 2]
        for g in range(8):
            d0, d1, d2 = _decompose(g, dims)
            self.assertEqual(g, d0 + 2 * d1 + 4 * d2)


class TestDimsSpanned(unittest.TestCase):
    dims = [2, 2, 2]

    def test_contiguous_half_spans_the_two_inner_dims(self):
        # ids 0-3 vary over d0 and d1 and are fixed at d2 = 0.
        self.assertEqual(_dims_spanned([0, 1, 2, 3], self.dims, "t"), [True, True, False])
        self.assertEqual(_dims_spanned([4, 5, 6, 7], self.dims, "t"), [True, True, False])

    def test_stride_four_group_spans_the_outer_dim(self):
        # {0,1,4,5} is fixed at d1 = 0 and varies over d0 and d2.
        self.assertEqual(_dims_spanned([0, 1, 4, 5], self.dims, "t"), [True, False, True])
        self.assertEqual(_dims_spanned([2, 3, 6, 7], self.dims, "t"), [True, False, True])

    def test_whole_machine_spans_everything(self):
        self.assertEqual(_dims_spanned(list(range(8)), self.dims, "t"), [True, True, True])

    def test_single_npu_spans_nothing(self):
        self.assertEqual(_dims_spanned([5], self.dims, "t"), [False, False, False])

    def test_partial_span_raises(self):
        # 2 of the 4 positions in dimension 0: ASTRA-Sim runs a collective over
        # the whole dimension, so this cannot be represented.
        with self.assertRaises(ValueError) as cm:
            _dims_spanned([0, 1], [4, 2], "t")
        self.assertIn("covers 2 of 4", str(cm.exception))

    def test_non_product_group_raises(self):
        with self.assertRaises(ValueError) as cm:
            _dims_spanned([0, 1, 2], self.dims, "t")
        self.assertIn("not a product of topology dimensions", str(cm.exception))


class TestValidateNetworkDims(unittest.TestCase):
    def test_product_must_match_npu_count(self):
        instances = [inst(4), inst(4)]
        self.assertEqual(_validate_network_dims([2, 2, 2], instances), [2, 2, 2])
        with self.assertRaises(ValueError) as cm:
            _validate_network_dims([2, 2], instances)
        self.assertIn("describes 4 NPU(s) but the cluster has 8", str(cm.exception))

    def test_prefill_instances_count_double(self):
        self.assertEqual(_validate_network_dims([2, 2], [inst(2, pd_type="prefill")]), [2, 2])

    def test_malformed_declarations_raise(self):
        instances = [inst(8)]
        for bad in ([], "2,2,2", [2, 0, 2], [2, True, 2], [2, 2.0, 2]):
            with self.assertRaises((TypeError, ValueError), msg=f"{bad!r} was accepted"):
                _validate_network_dims(bad, instances)


class TestBackwardCompatibility(unittest.TestCase):
    def test_absent_network_dims_keeps_the_inferred_topology(self):
        cases = [
            ([inst(4), inst(4)], [4, 2]),                       # independent replicas
            ([inst(8)], [8]),                                   # single instance
            ([inst(1) for _ in range(8)], [1, 8]),              # eight 1-GPU replicas
            ([inst(2, dp_group="A"), inst(2, dp_group="A")], [2, 2]),
        ]
        for instances, expected in cases:
            self.assertEqual(_compute_network_dims(instances), expected, msg=str(instances))
            self.assertEqual(_compute_network_dims(instances, None), expected)

    def test_declared_topology_wins(self):
        instances = [inst(4), inst(4)]
        self.assertEqual(_compute_network_dims(instances, [2, 2, 2]), [2, 2, 2])
        # trailing 1s are kept: a declaration is taken as written
        self.assertEqual(_compute_network_dims(instances, [8, 1]), [8, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
