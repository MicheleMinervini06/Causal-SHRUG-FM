import unittest

import numpy as np

from experiments import verification_scores as analysis


class VerificationScoreTests(unittest.TestCase):
    def test_halves_split_ties_and_control_for_events(self):
        stats = analysis.compare(np.array([0, 1, 1, 2]), np.array([1, 2, 4, 0]), np.array(["A", "A", "B", "B"]))
        self.assertEqual(stats["n_each_half"], 2)
        self.assertEqual(stats["low_mean"], 2)
        self.assertEqual(stats["high_mean"], 1.5)
        self.assertAlmostEqual(stats["within_event_rank_correlation"], 0)
        # The middle observation contributes to neither half for an odd count.
        odd = analysis.compare(np.arange(3), np.array([1, 99, 3]), np.array(["A", "A", "A"]))
        self.assertEqual(odd["high_minus_low"], 2)

    def test_missing_and_constant_scores_and_partial_resolution(self):
        rows = [{"a1": 1, "method": "proxy_only", "event_id": "A", "complete": i == 0, "n_resolved": 2-i%2,
                 **dict.fromkeys(analysis.SCORES, 0 if i < 2 else None),
                 **dict.fromkeys(analysis.ENDPOINTS, i)} for i in range(3)]
        contrasts = [r for r in analysis.associations(rows) if r["score"] == "average_entropy"
                     and r["method"] == "proxy_only" and r["a1"] == 1 and r["endpoint"] == "rmse_reduction"]
        complete, partial = contrasts
        self.assertEqual((complete["n_patches"], complete["missing_score"], complete["excluded_resolution"]), (1, 1, 1))
        self.assertEqual(partial["n_patches"], 2)
        self.assertIsNone(partial["spearman"])
        self.assertEqual(partial["high_minus_low"], 0)

    def test_join_preserves_id_and_decodes_direction(self):
        patch = dict(patch_id="p", event_id="A", original_split="test", reference_fraction=.2, predicted_fraction=.4)
        scored = {"p": {**patch, **dict.fromkeys(analysis.SCORES, 0)}}
        rows = [dict(**patch, a1=s, method=m, n_requested=1, n_resolved=1, complete="True",
                     rmse_change=-.1, bias_before=-.2, bias_after=.05, width95_change=.03,
                     coverage95_change=0, mean_absolute_effect_shift=.25)
                for s in (0, 1) for m in ("proxy_only", "with_scores")]
        meta = dict(protocol=dict(experiment="E06 single-patch verification", test_scores_sha256=analysis.e05.TEST_SCORES,
                                  posterior_sha256=analysis.e05.POSTERIOR), repeats=1, n_unresolved=0, n_incomplete_patch_groups=0)
        joined = analysis.join_impacts(scored, rows, meta)
        self.assertEqual(joined[0]["rmse_reduction"], .1)
        self.assertAlmostEqual(joined[0]["absolute_bias_reduction"], .15)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.join_impacts(scored, rows+[rows[0]], meta)
        with self.assertRaisesRegex(ValueError, "Missing"):
            analysis.join_impacts(scored, rows[1:], meta)

        burn_protocol = {"experiment": "E07 burn scars, single-patch verification", "strengths": [1.]}
        burn_meta = {**meta, "protocol": burn_protocol}
        confounded = [r for r in rows if r["a1"] == 1.]
        joined = analysis.join_impacts(scored, confounded, burn_meta, burn_protocol)
        self.assertEqual({r["method"] for r in joined}, {"proxy_only", "with_scores"})
        self.assertEqual({r["a1"] for r in analysis.associations(joined)}, {1.})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            analysis.join_impacts(scored, rows, burn_meta, burn_protocol)
