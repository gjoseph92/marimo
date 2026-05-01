# Copyright 2026 Marimo. All rights reserved.
"""Targeted tests for ``infer_kind`` — covers the table heuristics and a
couple of the trickier scalar cases. The exhaustive type matrix is left
to integration tests."""

from __future__ import annotations

import pytest

from marimo._dataflow.protocol import Kind
from marimo._dataflow.serialize import infer_kind


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([{"a": 1}, {"a": 2}], Kind.TABLE),  # records → table
        ([{"a": 1}], Kind.TABLE),  # single record is still a table
        ([], Kind.LIST),  # empty list is just a list
        ([1, 2, 3], Kind.LIST),  # scalars
        ([{"a": 1}, "not a dict"], Kind.LIST),  # heterogeneous → list
        (True, Kind.BOOLEAN),  # bool checked before int
        ({"a": 1}, Kind.DICT),  # plain dict, not a table
    ],
)
def test_infer_kind(value: object, expected: Kind) -> None:
    assert infer_kind(value) is expected


def test_infer_kind_pyarrow_table() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": [1, 2], "b": [3, 4]})
    assert infer_kind(table) is Kind.TABLE
    assert infer_kind(table.to_batches()[0]) is Kind.TABLE


def test_infer_kind_duckdb_relation() -> None:
    duckdb = pytest.importorskip("duckdb")
    rel = duckdb.sql("SELECT 1 AS a, 2 AS b UNION ALL SELECT 3, 4")
    assert infer_kind(rel) is Kind.TABLE


def test_infer_kind_polars_lazyframe_and_series() -> None:
    pl = pytest.importorskip("polars")
    assert infer_kind(pl.DataFrame({"a": [1]}).lazy()) is Kind.TABLE
    # Sibling types of polars are not tables — they shouldn't be coerced.
    assert infer_kind(pl.Series("a", [1, 2])) is Kind.ANY
