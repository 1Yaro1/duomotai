# -*- coding: utf-8 -*-

from __future__ import absolute_import

from typing import List


class FieldNames:

    MODEL_NAME = "model_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    ACTUAL_DATA = "actual_data"
    INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"
    ANOMALY_RATIO = "typical_anomaly_ratio"

    @classmethod
    def all_fields(cls) -> List[str]:
        return [
            cls.MODEL_NAME,
            cls.FILE_NAME,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.ACTUAL_DATA,
            cls.INFERENCE_DATA,
            cls.LOG_INFO,
        ]
