#!/usr/bin/env python3
"""involved_dim under an explicit physical topology when pp_size > 1.

A TP collective spans one pipeline stage's ranks. Deriving it from the
instance's whole NPU range charges the TP ALLREDUCE across the pipeline
dimension, which under a declared physical topology is a different link class.
These tests pin the scoping, the stage-agreement rule, and the pp=1 behaviour
that must not have moved.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from serving.core.config_builder import (  # noqa: E402
    _agreeing_dims, _resolve_dp_groups, _tp_group_ids,
)


def inst(num_npus, tp, pp, ep, start, iid, dg=None):
    return {"num_npus": num_npus, "tp_size": tp, "pp_size": pp, "ep_size": ep,
            "pd_type": None, "dp_group": dg, "instance_id": iid,
            "npu_start": start, "npu_span": num_npus, "is_moe": True}


class TestTpGroupIds(unittest.TestCase):
    def test_stage_ranges_match_the_converter(self):
        # llm_converter puts stage g on ids [g*tp, (g+1)*tp) of the instance.
        i = inst(4, tp=2, pp=2, ep=2, start=0, iid=0)
        self.assertEqual(_tp_group_ids(i, 0), [0, 1])
        self.assertEqual(_tp_group_ids(i, 1), [2, 3])
        j = inst(4, tp=2, pp=2, ep=2, start=4, iid=1)
        self.assertEqual(_tp_group_ids(j, 0), [4, 5])
        self.assertEqual(_tp_group_ids(j, 1), [6, 7])

    def test_pp1_is_the_whole_instance(self):
        i = inst(4, tp=4, pp=1, ep=4, start=0, iid=0)
        self.assertEqual(_tp_group_ids(i, 0), [0, 1, 2, 3])


class TestTp2Pp2(unittest.TestCase):
    """DP2 x PP2 x TP2/EP2 on 8 GPUs under network_dims [2,2,2]."""

    def setUp(self):
        self.insts = [inst(4, 2, 2, 2, start=0, iid=0, dg="A"),
                      inst(4, 2, 2, 2, start=4, iid=1, dg="A")]
        _resolve_dp_groups(self.insts, [2, 2, 2])

    def test_both_stages_derive_the_tp_dimension_only(self):
        # stage 0 = {0,1} and stage 1 = {2,3} both vary only in d0.
        for m in self.insts:
            self.assertEqual(m["tp_dim"], [True, False, False],
                             msg=f"instance {m['instance_id']}")

    def test_ep_stays_on_the_tp_dimension_when_it_fits(self):
        # ep_total (2) <= tp_size (2): experts do not spread past the instance.
        for m in self.insts:
            self.assertEqual(m["ep_dim"], [True, False, False])

    def test_pipeline_dimension_is_not_marked_involved(self):
        # d1 carries the PP hops (0->2, 1->3). No collective may claim it, or
        # the TP ALLREDUCE would be charged on the inter-node link.
        for m in self.insts:
            self.assertFalse(m["tp_dim"][1])
            self.assertFalse(m["ep_dim"][1])


class TestEpSpanningDp(unittest.TestCase):
    def test_ep_widens_to_the_dp_dimension_but_still_not_pp(self):
        # ep_total 4 > tp_size 2: the EP group is the DP x TP ranks of ONE
        # pipeline stage -- stage 0 is {0,1} + {4,5}, which varies in d0 and d2.
        insts = [inst(4, 2, 2, 4, start=0, iid=0, dg="A"),
                 inst(4, 2, 2, 4, start=4, iid=1, dg="A")]
        _resolve_dp_groups(insts, [2, 2, 2])
        for m in insts:
            self.assertEqual(m["tp_dim"], [True, False, False])
            self.assertEqual(m["ep_dim"], [True, False, True])
            self.assertFalse(m["ep_dim"][1], "PP dimension must stay uninvolved")


class TestStageAgreement(unittest.TestCase):
    def test_disagreeing_stages_raise(self):
        # A hypothetical placement whose stages sit on different dimensions has
        # no representation: the trace carries one involved_dim per instance.
        with self.assertRaises(ValueError) as cm:
            _agreeing_dims([[0, 1], [0, 2]], [2, 2, 2], "synthetic")
        self.assertIn("pipeline stages span different topology dimensions",
                      str(cm.exception))

    def test_agreeing_stages_pass_through(self):
        self.assertEqual(_agreeing_dims([[0, 1], [2, 3]], [2, 2, 2], "s"),
                         [True, False, False])


class TestPp1Unchanged(unittest.TestCase):
    def test_dp2_tp4_matches_the_placement_study(self):
        insts = [inst(4, 4, 1, 4, start=0, iid=0), inst(4, 4, 1, 4, start=4, iid=1)]
        _resolve_dp_groups(insts, [2, 2, 2])
        for m in insts:
            self.assertEqual(m["tp_dim"], [True, True, False])
            self.assertEqual(m["ep_dim"], [True, True, False])

    def test_dp4_tp2_matches_the_placement_study(self):
        insts = [inst(2, 2, 1, 2, start=2 * i, iid=i) for i in range(4)]
        _resolve_dp_groups(insts, [2, 2, 2])
        for m in insts:
            self.assertEqual(m["tp_dim"], [True, False, False])

    def test_no_network_dims_is_untouched(self):
        insts = [inst(4, 2, 2, 2, start=0, iid=0, dg="A"),
                 inst(4, 2, 2, 2, start=4, iid=1, dg="A")]
        _resolve_dp_groups(insts, None)
        # inferred topology is [tp, pp, dp]; dim 0 is TP by construction
        for m in insts:
            self.assertEqual(m["tp_dim"], [True, False, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
