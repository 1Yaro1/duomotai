"""Pre-registered GAIA DB baselines; training is gated by the external runner."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.MindTSWeb import MindTSWeb
from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
from ts_benchmark.baselines.MindTS.cmc_log_data import (
    BaselineLogView, BaselineWindowDataset, LogV2Store)


MODEL_SPECS = {
    "B0": {"baseline_mode": "metric_only_pure", "log_source": "unused"},
    "B1": {"baseline_mode": "metric_only_stats", "log_source": "unused"},
    "B2": {"baseline_mode": "metric_only_null_log", "log_source": "null"},
    "B3": {"baseline_mode": "metric_log_shifted", "log_source": "shifted"},
    "B4": {"baseline_mode": "metric_log_simple", "log_source": "aligned"},
}


class MindTSDBBaseline(MindTSWeb):
    """One trainer for B0--B4 with a frozen loss and data-order contract."""
    allowed_services = ("GAIA_dbservice1", "GAIA_dbservice2")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c = self.config
        if getattr(c, "baseline_id", None) not in MODEL_SPECS:
            raise ValueError("baseline_id must be one of B0--B4")
        spec = MODEL_SPECS[c.baseline_id]
        if c.baseline_mode != spec["baseline_mode"] or c.baseline_log_source != spec["log_source"]:
            raise ValueError("Baseline identity/source mapping drift")
        if c.contrastive_enabled or c.loss_mode != "reconstruction_ib":
            raise ValueError("B0--B4 use reconstruction + fixed information bottleneck only")
        if (c.log_semantic_dim, c.log_count_dim, c.log_shift_minutes) != (768, 12, 240):
            raise ValueError("Frozen log feature dimensions/shift changed")
        self.baseline_log_view = None

    def _prepare(self, train_data):
        c = self.config
        if len(train_data) != 5644 or not np.array_equal(train_data.index, np.arange(5644)):
            raise ValueError("Expected unchanged absolute training pool [0,5644)")
        self.detect_hyper_param_tune(train_data)
        c.task_name = "anomaly_detection"
        real_store = None
        if c.baseline_log_source in {"aligned", "shifted"}:
            real_store = LogV2Store(c.log_v2_cache_dir, c.log_v2_service)
        self.baseline_log_view = BaselineLogView(
            c.baseline_log_source, store=real_store,
            semantic_dim=c.log_semantic_dim, count_dim=c.log_count_dim,
            shift=c.log_shift_minutes)
        self.model = MINDTSModel(c).to(self.device)
        self.scaler.fit(train_data.iloc[:4515].values)

        def dataset(lo, hi):
            part = train_data.iloc[lo:hi]
            values = self.scaler.transform(part.values)
            return BaselineWindowDataset(values, lo, self.baseline_log_view)

        self.fit_dataset = dataset(0, 4515)
        self.val_dataset = dataset(4515, 5644)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable or any(parameter.requires_grad for parameter in self.model.model.parameters()):
            raise ValueError("Frozen Qwen/trainable parameter contract violated")
        self.optimizer = torch.optim.Adam(
            trainable, lr=c.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False)

    def scoring_dataset(self, scaled_values, absolute_start):
        if self.baseline_log_view is None:
            raise RuntimeError("Training preparation must establish the frozen log view")
        return BaselineWindowDataset(scaled_values, absolute_start, self.baseline_log_view)
