import csv
import os

import numpy as np

from ts_benchmark.baselines.MindTS.MindTS import MindTS as BaseMindTS


class MindTSLogged(BaseMindTS):
    """MindTS adapter that records each percentile threshold used at inference."""

    def detect_multi_label(self, test_data, test_text):
        percentile_calls = []
        original_percentile = np.percentile

        def recording_percentile(values, percentile, *args, **kwargs):
            threshold = original_percentile(values, percentile, *args, **kwargs)
            percentile_calls.append((float(percentile), float(threshold)))
            return threshold

        np.percentile = recording_percentile
        try:
            predictions, test_energy = super().detect_multi_label(test_data, test_text)
        finally:
            np.percentile = original_percentile

        ratios = list(predictions.keys())
        if len(ratios) != len(percentile_calls):
            raise RuntimeError(
                "Threshold recording mismatch: "
                f"{len(ratios)} ratios but {len(percentile_calls)} percentile calls"
            )

        threshold_log = os.environ.get("MINDTS_THRESHOLD_LOG_PATH")
        rows = []
        for ratio, (percentile, threshold) in zip(ratios, percentile_calls):
            row = {
                "anomaly_ratio": float(ratio),
                "percentile": percentile,
                "energy_threshold": threshold,
            }
            rows.append(row)
            print(
                "[threshold] "
                f"anomaly_ratio={row['anomaly_ratio']:.6g} "
                f"percentile={percentile:.6g} "
                f"energy_threshold={threshold:.17g}",
                flush=True,
            )

        if threshold_log:
            os.makedirs(os.path.dirname(os.path.abspath(threshold_log)), exist_ok=True)
            with open(threshold_log, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["anomaly_ratio", "percentile", "energy_threshold"],
                )
                writer.writeheader()
                writer.writerows(rows)

        return predictions, test_energy
