"""CPU integration of the real trainer using a tiny model, no GAIA access."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.MindTSWeb import MindTSWeb
from ts_benchmark.baselines.MindTS.contrastive_loss import WindowContrastHead
from ts_benchmark.baselines.MindTS.checkpointing import capture_rng, restore_rng, load_trusted


class TinyDataset:
    def __init__(self, first=0):
        self.first = first
    def __len__(self):
        return 19
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i + self.first)
        return (torch.randn(24, 2, generator=g), torch.randn(24, 2, generator=g),
                torch.ones(24), torch.zeros(24))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rec = torch.nn.Linear(2, 2)
        self.mask = torch.nn.Parameter(torch.zeros(1, 2, 1))
        self.metric = WindowContrastHead(24, 2, hidden=8, projection=4)
        self.log = WindowContrastHead(24, 2, hidden=8, projection=4)
    def forward(self, x, log, present, return_contrast=False):
        # Deliberately random even in eval, matching the historical custom gates.
        out = self.rec(x) + torch.rand_like(x) * .01
        logits = out[:, :2] @ log[:, :2].transpose(1, 2)
        result = (out, logits, logits.transpose(1, 2), self.mask.sigmoid().expand(len(x), -1, -1))
        if return_contrast:
            ma, mb = self.metric(x)
            la, lb = self.log(log)
            return (*result, dict(ma=ma, mb=mb, la=la, lb=lb))
        return result


class Harness(MindTSWeb):
    def __init__(self, directory, resume=None, evaluation=None, objective="dual"):
        self.config = SimpleNamespace(num_epochs=3, lr=.001, learning_rate=.001, lradj="type1",
            r=.5, lamda=.01, lamda1=.01, lamda2=.01, intra_weight=.1, inter_weight=.1,
            contrast_temperature=.2, contrast_min_negatives=1, patience=5,
            checkpoint_every=2, loss_mode=objective, resume_checkpoint=resume,
            evaluation_checkpoint=evaluation)
        self.ckpt_dir = Path(directory)
        self.identity = {"fixture": "trainer-resume-v1", "objective": objective}
        self.best_state = None
        self.device = torch.device("cpu")
        self.progress = dict(epoch=0, next_offset=0, order=[], global_step=0,
                             best_val=float("inf"), bad_epochs=0, stopped=False)
    def _prepare(self, unused):
        self.model = TinyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scaler = StandardScaler().fit(np.arange(20).reshape(10, 2))
        self.fit_dataset = TinyDataset()
        self.val_dataset = TinyDataset(4515)


def seed():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)


class ResumeTests(unittest.TestCase):
    def test_real_trainer_mid_epoch_resume(self):
        for objective in ("dual", "legacy_control"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                seed()
                full = Harness(root / "full", objective=objective)
                full.detect_multi_fit(None, object(), object())
                seed()
                broken = Harness(root / "interrupted", objective=objective)
                save = broken._save
                def stop(name):
                    save(name)
                    if name == "last" and broken.progress["global_step"] == 1:
                        raise InterruptedError("simulated crash after atomic checkpoint")
                broken._save = stop
                with self.assertRaises(InterruptedError):
                    broken.detect_multi_fit(None, object(), object())
                torch.rand(111)  # Simulate an unrelated process RNG before reconstruction.
                resumed = Harness(root / "resumed", root / "interrupted/last.pt", objective=objective)
                resumed.detect_multi_fit(None, object(), object())
                self.assertEqual(full.progress, resumed.progress)
                for key, value in full.model.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
                for key in ("torch_cpu",):
                    a = load_trusted(root / "full/last.pt")["rng"][key]
                    b = load_trusted(root / "resumed/last.pt")["rng"][key]
                    self.assertTrue(torch.equal(a, b))
                rows = [json.loads(s) for s in (root / "full/training.jsonl").read_text().splitlines()]
                self.assertEqual(sum(row["samples"] for row in rows), 19 * 3)
                # Check replay restore uses evaluation-start RNG, not constructor RNG.
                full._save("evaluation_start")
                x, log, present = full._batch(full.fit_dataset, [0, 1])
                full.model.eval()
                expected = full.model(x, log, present)[0]
                replay = Harness(root / "replay", evaluation=root / "full/evaluation_start.pt", objective=objective)
                replay.detect_multi_fit(None, object(), object())
                replay.model.eval()
                actual = replay.model(x, log, present)[0]
                self.assertTrue(torch.equal(expected, actual))

    def test_hidden_only_backbone_matches_causal_lm(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            attention_dropout=.1, output_hidden_states=True, output_attentions=True)
        config._attn_implementation = "eager"
        seed()
        model = Qwen2ForCausalLM(config)
        inputs = dict(input_ids=torch.arange(12).reshape(2, 6), attention_mask=torch.ones(2, 6),
                      output_hidden_states=True)
        for training in (False, True):
            model.train(training)
            rng = capture_rng()
            with torch.no_grad():
                expected = model(**inputs).hidden_states[-1]
                after = capture_rng()
                restore_rng(rng)
                actual = model.model(**inputs).hidden_states[-1]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after["torch_cpu"], capture_rng()["torch_cpu"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
