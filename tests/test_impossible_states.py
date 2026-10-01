import unittest
import numpy as np
from impossible_states.dataset import build_dataset, smoke_subset, validate
from impossible_states.analysis import auc, matched_vectors, text_metrics, unit

class DatasetTests(unittest.TestCase):
    def test_split_leakage_is_rejected(self):
        rows = build_dataset()
        validate(rows)
        validate(smoke_subset(rows))
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            validate(rows)

    def test_factorial_effects_are_recovered(self):
        rows = build_dataset()
        effects = {"neither": [0,0,0], "constipation": [2,0,0], "flatulence": [0,3,0],
                   "both": [2,3,5], "pain": [1,1,0], "discomfort": [0,0,1], "frustration": [0,1,0]}
        acts = np.array([[effects[r["condition"]]] for r in rows], dtype=float)
        vecs, _, _ = matched_vectors(acts, rows)
        np.testing.assert_allclose(vecs["constipation"][0], [2,0,2.5])
        np.testing.assert_allclose(vecs["flatulence"][0], [0,3,2.5])
        np.testing.assert_allclose(vecs["interaction"][0], [0,0,5])

    def test_auc_ties_and_direction(self):
        self.assertEqual(auc([1,1], [1,1]), 0.5)
        self.assertEqual(auc([2,3], [0,1]), 1.0)
        self.assertEqual(auc([0,1], [2,3]), 0.0)

    def test_word_boundaries(self):
        hits = text_metrics("My party has charts.")["keyword_hits"]
        self.assertEqual(hits["flatulence"], 0)
        self.assertEqual(hits["pain"], 0)
        self.assertEqual(text_metrics("I passed gas.")["keyword_hits"]["flatulence"], 1)

    def test_degenerate_directions_fail_explicitly(self):
        for v in ([0,0,0], [np.nan,1,2]):
            with self.assertRaises(ValueError):
                unit(np.array(v))

    def test_pronouns_do_not_corrupt_symptom_words(self):
        third = [r["text"] for r in build_dataset() if r["perspective"] == "third"]
        self.assertTrue(any("bowel movement" in t for t in third))
        self.assertFalse(any("movethement" in t for t in third))

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from impossible_states.engine import Engine
        torch.manual_seed(5)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=128, eos_token_id=None)
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs): return "prompt"
            def __call__(self, *args, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(input_ids=torch.tensor([[3,4,5]]))
            def decode(self, tokens, **kwargs): return " ".join(map(str,tokens))
        cls.engine = Engine(Qwen3ForCausalLM(config), Tokenizer(), "cpu")
        cls.direction = np.random.default_rng(3).normal(size=32)

    def run_arm(self, mode, forced=None):
        return self.engine.rollout("prompt", 0, self.direction, np.zeros(32), 1, 0.5,
            mode, max_tokens=8, pulse_tokens=2, forced_tokens=forced,
            readout=(2, self.direction, np.zeros(32), 1))

    def test_release_keeps_identical_prefix_and_cleans_cache(self):
        pulse = self.run_arm("pulse")
        rebuilt = self.run_arm("rebuild")
        self.assertEqual(pulse["generated_token_ids"][:2], rebuilt["generated_token_ids"][:2])
        forced = [6,7,8,9,10,11]
        baseline = self.run_arm("baseline", forced)
        rebuilt = self.run_arm("rebuild", forced)
        pulse = self.run_arm("pulse", forced)
        for b, r in zip(baseline["trace"][2:], rebuilt["trace"][2:]):
            self.assertAlmostEqual(b["downstream_projection"], r["downstream_projection"], places=5)
        self.assertTrue(any(abs(b["downstream_projection"]-p["downstream_projection"]) > 1e-6
                            for b,p in zip(baseline["trace"][2:], pulse["trace"][2:])))

    def test_zero_dose_does_not_change_generation(self):
        results = [self.engine.rollout("prompt", 0, self.direction, np.zeros(32), 1, 0,
                   mode, max_tokens=8, pulse_tokens=2) for mode in ("baseline", "continuous", "pulse", "rebuild", "perturb")]
        self.assertTrue(all(r["generated_token_ids"] == results[0]["generated_token_ids"] for r in results))
        self.assertEqual(len(self.engine.blocks[0]._forward_hooks), 0)

    def test_signed_displacement_and_release_schedule(self):
        for dose in (-0.2, 0.2):
            r = self.engine.rollout("prompt", 0, self.direction, np.zeros(32), 1,
                dose, "perturb", pulse_tokens=2, forced_tokens=[6,7,8,9,10])
            expected = [dose, dose, -dose, 0, 0]
            for trace, change in zip(r["trace"], expected):
                self.assertAlmostEqual(trace["after"]-trace["before"], change, places=5)

if __name__ == "__main__":
    unittest.main()
