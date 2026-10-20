import numpy as np
import pytest

from src.risk_scoring import probability_to_score, score_to_category


def test_known_values():
    assert list(probability_to_score([0, 0.3, 1])) == [0.0, 30.0, 100.0]


def test_scores_stay_in_bounds():
    scores = probability_to_score(np.linspace(0, 1, 101))
    assert scores.min() >= 0 and scores.max() <= 100


@pytest.mark.parametrize("bad", [[1.2], [-0.1], [float("nan")]])
def test_invalid_probability_raises(bad):
    with pytest.raises(ValueError):
        probability_to_score(bad)


def test_category_boundaries():
    assert score_to_category(0) == "low"
    assert score_to_category(29.9) == "low"
    assert score_to_category(30) == "medium"
    assert score_to_category(69.9) == "medium"
    assert score_to_category(70) == "high"
    assert score_to_category(100) == "high"


def test_category_out_of_range_raises():
    with pytest.raises(ValueError):
        score_to_category(101)


def test_custom_thresholds():
    assert score_to_category(50, {"medium": 10, "high": 40}) == "high"