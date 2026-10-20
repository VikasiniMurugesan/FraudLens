import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src.predict import load_bundle, score_transactions


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (X["a"] + X["b"] > 1).astype(int)
    pipe = Pipeline([("m", LogisticRegression())]).fit(X, y)
    return {"pipeline": pipe, "features": ["a", "b"], "threshold": 0.5}


def test_output_columns_and_bounds(bundle):
    df = pd.DataFrame({"a": [0.0, 2.0, -2.0], "b": [0.0, 2.0, -2.0]})
    out = score_transactions(df, bundle)
    assert {"risk_score", "risk_category", "flagged"} <= set(out.columns)
    assert out["risk_score"].between(0, 100).all()
    assert set(out["risk_category"]) <= {"low", "medium", "high"}


def test_extra_columns_are_kept(bundle):
    df = pd.DataFrame({"a": [0.0], "b": [0.0], "note": ["x"]})
    assert "note" in score_transactions(df, bundle).columns


def test_missing_column_raises(bundle):
    with pytest.raises(ValueError, match="Missing required columns"):
        score_transactions(pd.DataFrame({"a": [0.0]}), bundle)


def test_missing_values_raise(bundle):
    df = pd.DataFrame({"a": [np.nan], "b": [0.0]})
    with pytest.raises(ValueError, match="missing values"):
        score_transactions(df, bundle)


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained")
def test_real_model_loads_and_scores():
    bundle = load_bundle()
    row = {c: 0.0 for c in bundle["features"]}
    out = score_transactions(pd.DataFrame([row]), bundle)
    assert 0 <= out.loc[0, "risk_score"] <= 100