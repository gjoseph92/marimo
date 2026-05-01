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
