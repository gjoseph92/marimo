# Copyright 2026 Marimo. All rights reserved.
"""Tests for ``infer_kind`` and ``serialize_value``.

The serializer's table-shape pagination is the main thing under test. We
parametrize across every supported tabular library so a future refactor
can't quietly regress one of them.
"""

from __future__ import annotations

import pytest

from marimo._dataflow.protocol import Kind, VarView
from marimo._dataflow.serialize import infer_kind, serialize_value


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


N_ROWS = 250


def _all_table_cases() -> dict[str, object]:
    pa = pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    pd = pytest.importorskip("pandas")

    rows = list(range(N_ROWS))
    return {
        "pandas": pd.DataFrame({"a": rows}),
        "polars": pl.DataFrame({"a": rows}),
        "polars-lazy": pl.DataFrame({"a": rows}).lazy(),
        "pyarrow": pa.table({"a": rows}),
        "pyarrow-batch": pa.table({"a": rows}).to_batches()[0],
        "duckdb": duckdb.sql(
            "SELECT * FROM (VALUES "
            + ", ".join(f"({i})" for i in rows)
            + ") t(a)"
        ),
    }


def test_no_view_returns_full_table() -> None:
    """The default — no ``view`` — must serialize every row.

    The previous behavior silently truncated to a 100-row preview which
    made the ``json`` channel useless for actual data fetching. Lock that
    in across every supported library.
    """
    for label, value in _all_table_cases().items():
        out, _ = serialize_value(value, encoding="json")
        assert isinstance(out, list)
        assert len(out) == N_ROWS, label
        # First and last rows survived the round-trip.
        assert out[0]["a"] == 0, label
        assert out[-1]["a"] == N_ROWS - 1, label


def test_view_row_limit_caps_each_table_type() -> None:
    """An explicit ``rowLimit`` truncates uniformly across libraries."""
    view = VarView(row_limit=10)
    for label, value in _all_table_cases().items():
        out, _ = serialize_value(value, encoding="json", view=view)
        assert len(out) == 10, label
        assert out[0]["a"] == 0, label
        assert out[-1]["a"] == 9, label


def test_view_row_offset_paginates_each_table_type() -> None:
    """``rowOffset`` + ``rowLimit`` give callers stateless pagination."""
    view = VarView(row_limit=10, row_offset=20)
    for label, value in _all_table_cases().items():
        out, _ = serialize_value(value, encoding="json", view=view)
        assert len(out) == 10, label
        assert out[0]["a"] == 20, label
        assert out[-1]["a"] == 29, label


def test_view_offset_only_returns_remaining_rows() -> None:
    """``rowOffset`` without ``rowLimit`` skips the prefix and keeps the rest.

    DuckDB uses a sentinel limit since its API requires one; the rest use
    native open-ended slicing. Verify both paths behave the same way.
    """
    view = VarView(row_offset=N_ROWS - 5)
    for label, value in _all_table_cases().items():
        out, _ = serialize_value(value, encoding="json", view=view)
        assert len(out) == 5, label
        assert out[0]["a"] == N_ROWS - 5, label
        assert out[-1]["a"] == N_ROWS - 1, label


def test_view_paginates_top_level_list_of_records() -> None:
    """List-of-records is reported as ``Kind.TABLE``, so it paginates too.

    This is the path the inspector hits when the notebook produces a
    plain ``list[dict]`` rather than a real DataFrame. Without it, the
    inspector's per-page fetch would fall through to the generic list
    branch and silently ignore ``rowLimit`` / ``rowOffset``.
    """
    rows = [{"i": i} for i in range(N_ROWS)]
    out, _ = serialize_value(
        rows, encoding="json", view=VarView(row_limit=10, row_offset=20)
    )
    assert isinstance(out, list)
    assert len(out) == 10
    assert out[0] == {"i": 20}
    assert out[-1] == {"i": 29}


def test_view_does_not_paginate_nested_tables() -> None:
    """Nested tabular values inside lists/dicts aren't paginated.

    The view applies to the top-level subscribed value; nested tables
    flow through ``_to_json`` recursion without view propagation. Lock
    that in so the contract is "if you want pagination, subscribe to the
    table directly."
    """
    pd = pytest.importorskip("pandas")
    nested = {"frame": pd.DataFrame({"a": list(range(N_ROWS))})}
    out, _ = serialize_value(
        nested, encoding="json", view=VarView(row_limit=10)
    )
    assert isinstance(out, dict)
    assert len(out["frame"]) == N_ROWS
