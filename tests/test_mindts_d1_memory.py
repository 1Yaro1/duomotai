"""D1 scope and eager/no-retention equivalence. No GAIA metrics or training."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ts_benchmark.baselines.MindTS.models.MindTS_model import frozen_prompt_last_hidden
from ts_benchmark.baselines.MindTS.checkpointing import capture_rng, restore_rng
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM


def assert_pair(reference, lean, input_ids, attention_mask):
    for training in (False, True):
        reference.train(training)
        lean.train(training)
        before = capture_rng()
        with torch.no_grad():
            output = reference.model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True, output_attentions=True,
                                     use_cache=True, return_dict=True)
            expected = output.hidden_states[-1].clone()
            del output
        after = capture_rng()
        restore_rng(before)
        actual = frozen_prompt_last_hidden(lean, input_ids, attention_mask)
        assert torch.equal(expected, actual), f"Hidden states differ (train={training}, max={float((expected-actual).abs().max())})"
        assert torch.equal(after["torch_cpu"], torch.get_rng_state())
        for old, new in zip(after["torch_cuda"], torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []):
            assert torch.equal(old, new), "CUDA RNG differs"
        assert not actual.requires_grad


class D1Tests(unittest.TestCase):
    def test_scope_and_configuration(self):
        from scripts.autoresearch import gaia_d1_b8_runner as d1, gaia_web_b8_runner as web
        from ts_benchmark.baselines.MindTSD1 import MindTSD1
        from ts_benchmark.baselines.MindTSWeb import MindTSWeb
        candidate, protocol, baseline = d1.configure()
        self.assertEqual(d1.SERVICES, ["GAIA_dbservice1", "GAIA_dbservice2"])
        self.assertEqual([s["name"] for s in protocol["services"]], d1.SERVICES)
        self.assertEqual(tuple(d1.SERVICES), MindTSD1.allowed_services)
        self.assertEqual(tuple(web.SERVICES), MindTSWeb.allowed_services)
        self.assertTrue(candidate["llm_memory_efficient"])
        self.assertEqual(candidate["batch_size"], 8)
        self.assertEqual(baseline["batch_size"], 1)  # Historical config not overwritten.
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        d1.frozen_source_guard()

    def test_two_service_mean_and_missing_failure(self):
        from scripts.autoresearch import gaia_d1_b8_runner as d1
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            protocol = {"services": [{"name": s} for s in d1.SERVICES]}
            for service, f1 in zip(d1.SERVICES, (.6, .8)):
                row = {key: .5 for key in d1.frozen.METRICS}
                row.update(service=service, F1=f1, energy_threshold=.3)
                (directory / (service + ".best.json")).write_text(json.dumps(row))
            d1.summarize(directory, protocol)
            self.assertEqual((directory / "summary.log").read_text().splitlines()[-1], '{"mean_best_f1": 0.7000}')
            (directory / (d1.SERVICES[1] + ".best.json")).unlink()
            with self.assertRaises(FileNotFoundError):
                d1.summarize(directory, protocol)

    def test_cpu_sdpa_fallback_vs_eager(self):
        self.compare_tiny("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_cuda_sdpa_fallback_vs_eager(self):
        self.compare_tiny("cuda")

    def compare_tiny(self, device):
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            attention_dropout=.1)
        config._attn_implementation = "sdpa"
        reference = Qwen2ForCausalLM(config).to(device)
        eager = copy.deepcopy(config)
        eager._attn_implementation = "eager"
        lean = Qwen2ForCausalLM(eager).to(device)
        lean.load_state_dict(reference.state_dict(), strict=True)
        ids = torch.arange(48, device=device).reshape(4, 12)
        mask = torch.ones_like(ids)
        mask[1, :3] = 0  # Same padded inputs, no change in mask semantics.
        torch.use_deterministic_algorithms(True)
        assert_pair(reference, lean, ids, mask)
        with self.assertRaises(ValueError):
            frozen_prompt_last_hidden(reference, ids, mask)


def check_real_llm():
    if not torch.cuda.is_available():
        raise RuntimeError("Real LLM comparison requires the GPU")
    path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config = AutoConfig.from_pretrained(path)
    config.num_hidden_layers = 6
    config.output_attentions = True
    config.output_hidden_states = True
    reference = AutoModelForCausalLM.from_pretrained(path, config=config, attn_implementation="sdpa").cuda()
    lean = AutoModelForCausalLM.from_config(copy.deepcopy(config), attn_implementation="eager").cuda()
    lean.load_state_dict(reference.state_dict(), strict=True)
    # Synthetic IDs only, not training/validation/test examples. Full length128.
    ids = torch.arange(8*128, device="cuda").reshape(8, 128) % config.vocab_size
    mask = torch.ones_like(ids)
    mask[1, :17] = 0
    torch.manual_seed(2021)
    torch.use_deterministic_algorithms(True)
    assert_pair(reference, lean, ids, mask)
    print(json.dumps({"real_frozen_llm_equivalence": "exact_hidden_states_and_rng",
                      "layers": 6, "prompt_batch": 8, "tokens": 128,
                      "modes": ["eval", "train"], "dtype": str(next(lean.parameters()).dtype),
                      "training_updates": 0, "test_used": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-llm-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(D1Tests))
    if not results.wasSuccessful():
        sys.exit(1)
    if args.real_llm_check:
        check_real_llm()
