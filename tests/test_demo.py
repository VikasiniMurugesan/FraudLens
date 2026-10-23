import pandas as pd
import pytest

import config
from src.demo import find_column, load_demo
from src.predict import score_transactions


def test_find_column_returns_first_match():
    df = pd.DataFrame({"is_fraud": [0], "Class": [1]})
    assert find_column(df, ["Class", "is_fraud"]) == "Class"


def test_find_column_returns_none_when_absent():
    assert find_column(pd.DataFrame({"x": [1]}), ["Class", "is_fraud"]) is None


@pytest.mark.skipif(not config.DEMO_MODEL_PATH.exists(), reason="demo model not trained")
def test_demo_loads_and_scores():
    bundle, df = load_demo()
    scored = score_transactions(df, bundle)
    assert len(scored) > 0
    assert scored["risk_score"].between(0, 100).all()