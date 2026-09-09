"""CPU/CUDA prompt-value equivalence. Optional GAIA gradient-training-only check."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Load the production helper without constructing/importing an external LLM.
source = ROOT / "ts_benchmark/baselines/MindTS/models/MindTS_model.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompt_median_values")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
prompt_median_values = namespace["prompt_median_values"]


def compare_cuda(patches):
    """Temporarily allow the ORIGINAL operator for reference, in this test only."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        reference = torch.median(patches, dim=2).values.detach().cpu()
        torch.use_deterministic_algorithms(True)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        actual = prompt_median_values(patches)
        assert torch.equal(reference, actual), "Median values differ"
        assert str(reference.tolist()) == str(actual.tolist()), "Prompt median strings differ"
        assert not actual.requires_grad and actual.device.type == "cpu"
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
        assert torch.are_deterministic_algorithms_enabled()
        return actual.numel()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class PromptMedianTests(unittest.TestCase):
    def test_even_lower_not_average(self):
        x = torch.tensor([[[1., 9., 3., 7., 2., 8.]]])
        self.assertEqual(prompt_median_values(x).item(), 3.)

    def test_odd_and_duplicate_values(self):
        x = torch.tensor([[[4., 1., 2., 2., 8.], [-5., -1., -3., -3., 0.]]])
        self.assertTrue(torch.equal(prompt_median_values(x), torch.tensor([[2., -3.]])))

    def test_detach_preserve_input_and_cpu_rng(self):
        x = torch.arange(48.).reshape(2, 4, 6).requires_grad_()
        before, rng = x.detach().clone(), torch.get_rng_state()
        output = prompt_median_values(x)
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA reference requires GPU")
    def test_cuda_reference_and_prompt_text(self):
        generator = torch.Generator().manual_seed(2021)
        values = torch.randn(128, 4, 6, generator=generator)
        ties = torch.tensor([0., 1., 1., 1., 2., 2.]).expand(32, 4, 6)
        for fixture in (values, ties, torch.zeros(8, 4, 6)):
            compare_cuda(fixture.cuda().requires_grad_())


def check_gaia_training():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; do not silently skip GAIA CUDA comparison")
    from sklearn.preprocessing import StandardScaler
    from ts_benchmark.baselines.MindTS.cmc_log_data import load_numeric_prefix
    protocol = json.loads((ROOT / "config/autoresearch/gaia_protocol.json").read_text())
    selected = [s for s in protocol["services"] if s["name"] in ("GAIA_webservice1", "GAIA_webservice2")]
    assert len(selected) == 2
    torch.use_deterministic_algorithms(True)
    for service in selected:
        frame = load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"],
                                    service["channels"], include_label=False).iloc[:4515]
        scaler = StandardScaler().fit(frame.values)
        values = torch.tensor(scaler.transform(frame.values), dtype=torch.float32, device="cuda")
        windows, medians = len(frame) - 24 + 1, 0
        for offset in range(0, windows, 8):
            x = torch.stack([values[s:s+24] for s in range(offset, min(offset+8, windows))])
            # Match the actual GPU prompt path, including normalization and float32.
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            patches = x.permute(0, 2, 1).contiguous().reshape(-1, 24).unfold(1, 6, 6)
            medians += compare_cuda(patches)
        print(json.dumps({"prompt_median_equivalence": "exact_values_and_strings",
                          "service": service["name"], "training_windows": windows,
                          "median_values": medians, "training_updates": 0,
                          "validation_windows_used": 0, "test_used": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-training-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PromptMedianTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if args.gaia_training_check:
        check_gaia_training()
