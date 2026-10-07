"""
Converts raw model output into a `Signal` the risk/execution layers can act
on, without those layers needing to know anything about tensors, softmax,
or model internals. Keeping this boundary explicit means the model can be
swapped, retrained, or even replaced by an ensemble of models without
touching risk or execution code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass
class Signal:
    ticker: str
    timestamp: object
    action: Action
    confidence: float          # softmax probability of the chosen action, in [0, 1]
    expected_return: float     # regression head's forward-return estimate
    model_version: str


def to_signal(ticker: str, timestamp, action: int, confidence: float,
              expected_return: float, model_version: str) -> Signal:
    return Signal(
        ticker=ticker,
        timestamp=timestamp,
        action=Action(action),
        confidence=float(confidence),
        expected_return=float(expected_return),
        model_version=model_version,
    )
