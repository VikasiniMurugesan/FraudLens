import io

import pytest

from src.data_loader import read_transactions_csv

REQUIRED = ["a", "b"]


def read(text: str, **kwargs):
    return read_transactions_csv(io.StringIO(text), REQUIRED, **kwargs)


def test_valid_file_loads():
    df = read("a,b\n1,2\n3,4\n")
    assert len(df) == 2


def test_empty_file_raises():
    with pytest.raises(ValueError, match="empty"):
        read("")


def test_header_only_raises():
    with pytest.raises(ValueError, match="no rows"):
        read("a,b\n")


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        read("a,c\n1,2\n")


def test_text_in_number_column_raises():
    with pytest.raises(ValueError, match="Non-numeric"):
        read("a,b\n1,x\n")


def test_missing_value_raises():
    with pytest.raises(ValueError, match="missing values"):
        read("a,b\n1,\n2,3\n")


def test_too_many_rows_raises():
    with pytest.raises(ValueError, match="limit"):
        read("a,b\n1,2\n3,4\n5,6\n", max_rows=2)


def test_negative_amount_raises():
    with pytest.raises(ValueError, match="negative"):
        read_transactions_csv(io.StringIO("Amount\n-5\n"), ["Amount"])