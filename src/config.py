"""Tiny config loader: reads config.yaml into the dataclasses each module
already defines, so there's exactly one place (config.yaml) to tune
behavior without touching code."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.model.network import ModelConfig
from src.data.features import LabelConfig
from src.training.trainer import TrainerConfig
from src.risk.manager import RiskConfig
from src.training.drift import DriftConfig


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    n_features = 15  # len(FEATURE_COLUMNS) in src/data/features.py
    raw["_model_cfg"] = ModelConfig(n_features=n_features, **raw["model"])
    raw["_label_cfg"] = LabelConfig(**raw["label"])
    raw["_trainer_cfg"] = TrainerConfig(**raw["trainer"])
    raw["_risk_cfg"] = RiskConfig(**raw["risk"])
    raw["_drift_cfg"] = DriftConfig(**raw["drift"])
    return raw
