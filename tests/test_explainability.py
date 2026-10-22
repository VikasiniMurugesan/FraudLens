import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.explainability import explain_transaction, reason_to_text


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"] > 1).astype(int)  # only feature "a" matters
    pipe = Pipeline([("m", LogisticRegression())]).fit(X, y)
    return {"pipeline": pipe, "features": ["a", "b"], "threshold": 0.5,
            "reference": X.median().to_dict()}


def test_driving_feature_is_top_reason(bundle):
    row = pd.Series({"a": 3.0, "b": 0.0})
    reasons = explain_transaction(row, bundle)
    assert reasons and reasons[0]["feature"] == "a"
    assert reasons[0]["points"] > 0


def test_typical_transaction_has_no_reasons(bundle):
    row = pd.Series(bundle["reference"])
    assert explain_transaction(row, bundle) == []


def test_top_n_limits_output(bundle):
    row = pd.Series({"a": 3.0, "b": 3.0})
    assert len(explain_transaction(row, bundle, top_n=1)) <= 1


def test_missing_reference_raises(bundle):
    del bundle["reference"]
    with pytest.raises(ValueError, match="reference"):
        explain_transaction(pd.Series({"a": 1.0, "b": 1.0}), bundle)


def test_missing_column_raises(bundle):
    with pytest.raises(ValueError, match="Missing required columns"):
        explain_transaction(pd.Series({"a": 1.0}), bundle)


def test_text_mentions_feature_and_direction():
    text = reason_to_text({"feature": "a", "value": 3.0, "typical": 0.0, "points": 40.0})
    assert "a is higher than typical" in text