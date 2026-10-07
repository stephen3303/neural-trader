import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.execution.alpaca_broker import AlpacaBroker


def test_refuses_to_construct_outside_paper_mode():
    # This guard is the whole point of the class: it must be impossible to
    # end up with a live-money AlpacaBroker by accident (wrong default,
    # typo'd config, copy-pasted example). The check has to happen before
    # any network call or real `alpaca` client construction.
    with pytest.raises(ValueError, match="paper"):
        AlpacaBroker(api_key="x", secret_key="y", paper=False)


def test_constructs_in_paper_mode():
    broker = AlpacaBroker(api_key="x", secret_key="y", paper=True)
    assert broker.client is not None
