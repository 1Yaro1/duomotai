"""Versioned web-only training; inherited label scoring/threshold protocol is unchanged."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, default_collate

from ts_benchmark.baselines.MindTSLogged import MindTSLogged
from ts_benchmark.baselines.MindTS.MindTS import Bottleneck_loss, clip_loss
from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
from ts_benchmark.baselines.MindTS.cmc_log_data import LogV2Dataset
from ts_benchmark.baselines.MindTS.contrastive_loss import dual_contrast
from ts_benchmark.baselines.MindTS.checkpointing import (
    atomic_save, capture_rng, restore_rng, cpu_tree, load_trusted, epoch_order)
from ts_benchmark.baselines.MindTS.utils.tools import adjust_learning_rate


class MindTSWeb(MindTSLogged):
    allowed_services = ("GAIA_webservice1", "GAIA_webservice2")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c = self.config
        if c.log_v2_service not in self.allowed_services:
            raise ValueError("Service outside this trainer's approved two-instance scope")
        if c.batch_size != 8 or c.seq_len != 24 or c.num_epochs != 3:
            raise ValueError("Expected approved batch=8, window=24, epochs=3")
        if c.parallel_strategy not in (None, "DP") or torch.cuda.device_count() > 1:
            raise ValueError("Expose exactly one GPU; multi-device resume is not implemented")
        self.ckpt_dir = Path(c.checkpoint_dir)
        self.identity = c.checkpoint_identity
        self.best_state = None
        self.progress = dict(epoch=0, next_offset=0, order=[], global_step=0,
                             best_val=math.inf, bad_epochs=0, stopped=False)

    def _prepare(self, train_data):
        if len(train_data) != 5644 or not np.array_equal(train_data.index, np.arange(5644)):
            raise ValueError("Expected unchanged absolute training pool [0,5644)")
        self.detect_hyper_param_tune(train_data)
        self.config.task_name = "anomaly_detection"
        self.model = MINDTSModel(self.config).to(self.device)
        self.scaler.fit(train_data.iloc[:4515].values)
        def dataset(lo, hi, mode):
            part = train_data.iloc[lo:hi]
            frame = pd.DataFrame(self.scaler.transform(part.values), index=part.index, columns=part.columns)
            return LogV2Dataset(frame, self.log_v2_store, mode=mode)
        self.fit_dataset = dataset(0, 4515, "train")
        self.val_dataset = dataset(4515, 5644, "val")
        # This loader is only used by the inherited threshold-scoring path.
        self.train_data_loader = DataLoader(self.fit_dataset, batch_size=8, shuffle=True,
                                           num_workers=0, drop_last=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _batch(self, dataset, indices):
        values = default_collate([dataset[i] for i in indices])
        return tuple(x.float().to(self.device) for x in values[:3])

    def _payload(self, kind):
        return {"schema_version": 1, "kind": kind, "identity": self.identity,
                "config": vars(self.config).copy(), "model": cpu_tree(self.model.state_dict()),
                "optimizer": cpu_tree(self.optimizer.state_dict()),
                "scheduler": {"implementation": "adjust_learning_rate", "epoch": self.progress["epoch"]},
                "progress": self.progress.copy(), "scaler": vars(self.scaler).copy(),
                "best_model": self.best_state if kind == "last" else None,
                "rng": capture_rng()}

    def _save(self, name):
        atomic_save(self._payload(name), self.ckpt_dir / (name + ".pt"))

    def _restore(self, path, *, evaluation=False):
        state = load_trusted(path, self.identity)
        if evaluation and state["kind"] != "evaluation_start":
            raise ValueError("Exact score replay requires evaluation_start.pt, not an arbitrary training checkpoint")
        if not evaluation and state["kind"] not in ("last", "best"):
            raise ValueError("Resume requires last.pt or best.pt")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.__dict__.update(state["scaler"])
        self.progress = state["progress"]
        self.best_state = state.get("best_model")
        if state["kind"] in ("best", "evaluation_start"):
            self.best_state = state["model"]
        order = self.progress["order"]
        if order and sorted(order) != list(range(len(self.fit_dataset))):
            raise ValueError("Checkpoint sampler order is not the complete training-window permutation")
        if not 0 <= self.progress["next_offset"] <= len(self.fit_dataset):
            raise ValueError("Invalid checkpoint sampler cursor")
        restore_rng(state["rng"])

    @torch.no_grad()
    def _validation_mse(self):
        # New and paired-control modes use the same explicitly documented early-stop rule.
        c = self.config
        rng = capture_rng()
        try:
            torch.manual_seed(getattr(c, "seed", 2021))
            self.model.eval()
            total, count = 0.0, 0
            for offset in range(0, len(self.val_dataset), 8):
                ids = list(range(offset, min(offset + 8, len(self.val_dataset))))
                x, log, present = self._batch(self.val_dataset, ids)
                out = self.model(x, log, present)[0]
                total += float((out - x).square().sum().item())
                count += x.numel()
            result = total / count
            if not math.isfinite(result):
                raise ValueError("Non-finite validation reconstruction MSE")
            return result
        finally:
            restore_rng(rng)
            self.model.train()

    def detect_multi_fit(self, train_data, train_text, train_label):
        del train_text, train_label  # Never use real anomaly labels for optimization/pairing.
        self._prepare(train_data)
        if getattr(self.config, "evaluation_checkpoint", None):
            self._restore(self.config.evaluation_checkpoint, evaluation=True)
            self.early_stopping = SimpleNamespace(check_point=self.best_state)
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=False)
        if getattr(self.config, "resume_checkpoint", None):
            self._restore(self.config.resume_checkpoint)
        c = self.config
        contrastive_enabled = getattr(c, "contrastive_enabled", True)
        log_path = self.ckpt_dir / "training.jsonl"
        while self.progress["epoch"] < c.num_epochs and not self.progress["stopped"]:
            epoch = self.progress["epoch"]
            if not self.progress["order"]:
                self.progress["order"] = epoch_order(
                    len(self.fit_dataset), epoch, getattr(c, "seed", 2021))
            self.model.train()
            while self.progress["next_offset"] < len(self.fit_dataset):
                offset = self.progress["next_offset"]
                if self.progress["global_step"] >= getattr(c, "max_optimizer_updates", math.inf):
                    raise RuntimeError("Maximum optimizer updates exceeded; no automatic retry")
                ids = self.progress["order"][offset:offset + 8]
                x, log, present = self._batch(self.fit_dataset, ids)
                self.optimizer.zero_grad(set_to_none=True)
                if contrastive_enabled:
                    out, lt, ll, mask, views = self.model(x, log, present, return_contrast=True)
                else:
                    out, lt, ll, mask = self.model(x, log, present, return_contrast=False)
                    views = None
                rec = (out - x).square().mean()
                ib = Bottleneck_loss(mask, c.r, c.lamda)
                if contrastive_enabled:
                    starts = torch.tensor(ids, device=self.device, dtype=torch.long) + self.fit_dataset.first
                    terms = dual_contrast(views, starts, present, c.contrast_temperature, c.contrast_min_negatives)
                else:
                    zero = rec.detach().new_zeros(())
                    terms = dict(intra_metric=zero, intra_log=zero, inter=zero,
                                 valid_metric_anchors=0, valid_log_anchors=0,
                                 valid_inter_anchors=0, negative_metric_mean=0.0,
                                 negative_log_mean=0.0)
                if c.loss_mode == "dual" and contrastive_enabled:
                    loss = rec + c.lamda2 * ib + c.intra_weight * (terms["intra_metric"] + terms["intra_log"]) / 2 + c.inter_weight * terms["inter"]
                elif c.loss_mode == "legacy_control" and contrastive_enabled:
                    # Same new-mode architecture/RNG/batching/early-stop/checkpoint plumbing.
                    # Only objective differs; preserve old summed CLIP for this control.
                    loss = rec + c.lamda2 * ib + c.lamda1 * clip_loss(lt, ll)
                elif c.loss_mode == "reconstruction_ib" and not contrastive_enabled:
                    loss = rec + c.lamda2 * ib
                else:
                    raise ValueError("Unsupported objective")
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training objective; no retry")
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise ValueError("Non-finite gradient: " + name)
                self.optimizer.step()
                self.progress["global_step"] += 1
                self.progress["next_offset"] += len(ids)
                row = {"event": "step", "epoch": epoch, "step": self.progress["global_step"],
                       "samples": len(ids), "loss": float(loss.detach()), "rec": float(rec.detach()),
                       "ib": float(ib.detach()), **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in terms.items()}}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                if self.progress["global_step"] % 20 == 0:
                    print(json.dumps(row), flush=True)
                del out, lt, ll, mask, views, terms, loss, rec, ib
                if self.progress["global_step"] == 1 or self.progress["global_step"] % c.checkpoint_every == 0:
                    self._save("last")
            val = self._validation_mse()
            improved = val < self.progress["best_val"]
            if improved:
                self.progress["best_val"] = val
                self.best_state = cpu_tree(self.model.state_dict())
                self.progress["bad_epochs"] = 0
            else:
                self.progress["bad_epochs"] += 1
            self.progress["stopped"] = self.progress["bad_epochs"] >= c.patience
            adjust_learning_rate(self.optimizer, epoch + 1, c)
            self.progress.update(epoch=epoch + 1, next_offset=0, order=[])
            if improved:
                self._save("best")
            self._save("last")
            print(json.dumps({"event": "epoch_complete", "epoch": epoch + 1, "validation_mse": val,
                              "best_validation_mse": self.progress["best_val"]}), flush=True)
        if self.best_state is None:
            raise RuntimeError("No completed valid epoch/checkpoint")
        self.model.load_state_dict(self.best_state)
        self.early_stopping = SimpleNamespace(check_point=self.best_state)

    @torch.no_grad()
    def detect_multi_label(self, data, text):
        # Preserve old random masking/gating, but checkpoint the EXACT pre-scoring RNG.
        # No gradient graph is needed during scoring; mathematical operations are unchanged.
        if getattr(self.config, "evaluation_checkpoint", None):
            self._restore(self.config.evaluation_checkpoint, evaluation=True)
        else:
            self._save("evaluation_start")
        return super().detect_multi_label(data, text)
