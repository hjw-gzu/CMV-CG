"""Boundary cases that affect paper captions, beam lengths and fixed-pool selection."""
import unittest
from cmv_cg.io import ROOT, read
from cmv_cg.replay import KEYS, select
from cmv_cg.search import beam_search


class TestProtocol(unittest.TestCase):
    def test_completed_beam_keeps_score_and_length(self):
        transitions = {(): [(13, -0.4), (1, -0.5)], (1,): [(13, -0.1)]}
        def model(prefixes):
            return [transitions[p] for p in prefixes]
        beams = beam_search(model, beam_size=2, max_tokens=4, stop_token=13)
        self.assertEqual([r["tokens"] for r in beams], [[13], [1, 13]])
        self.assertEqual([r["length"] for r in beams], [1, 2])
        self.assertAlmostEqual(beams[1]["mean_logprob"], -0.3)
        self.assertTrue(all(r["ended"] for r in beams))

    def test_stable_tie_and_constant_score(self):
        def scored(values):
            return {"raw_scores": dict(zip(KEYS, values))}
        pool = [scored([1, 3, 0, 0]), scored([1, 2, 1, 0])]
        self.assertEqual(select(pool, [1, 0, 0, 0]), 0)
        self.assertEqual(select(pool, [0, 0, 1, 0]), 1)
        self.assertEqual(select(pool, [0, .5, .5, 0]), 0)

    def test_data_cap_and_invalid_scores(self):
        beams = beam_search(lambda prefixes: [[(1, -1)] for _ in prefixes], beam_size=1, max_tokens=2)
        self.assertEqual(beams[0]["tokens"], [1, 1])
        self.assertFalse(beams[0]["ended"])
        with self.assertRaises(ValueError):
            select([{"raw_scores": dict.fromkeys(KEYS, float("nan"))}])

    def test_fig1_outputs(self):
        data = read(ROOT / "examples/motivation.json")
        for panel in data["panels"].values():
            index = select(panel["candidates"], panel["weights"])
            self.assertEqual(panel["candidates"][index]["caption"], panel["caption"])


if __name__ == "__main__":
    unittest.main()

