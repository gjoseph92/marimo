# Copyright 2026 Marimo. All rights reserved.
"""Serialize Python values for the dataflow wire protocol."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from marimo._dataflow.protocol import Kind

if TYPE_CHECKING:
    from marimo._dataflow.protocol import VarView


def infer_kind(value: Any) -> Kind:
    """Infer the Kind for a Python value."""
    if value is None:
        return Kind.NULL
    if isinstance(value, bool):
        return Kind.BOOLEAN
    if isinstance(value, int):
        return Kind.INTEGER
    if isinstance(value, float):
        return Kind.NUMBER
    if isinstance(value, str):
        return Kind.STRING
    if isinstance(value, bytes):
        return Kind.BYTES
    if isinstance(value, list):
        # A non-empty list of dicts is the JSON-records shape of a
        # table; surface it as TABLE so the client picks the table
        # renderer. Sample a bounded prefix to stay cheap on large
        # inputs.
        if value and all(
            isinstance(v, dict) for v in value[:_TABLE_SAMPLE_SIZE]
        ):
            return Kind.TABLE
        return Kind.LIST
    if isinstance(value, dict):
        return Kind.DICT
    if isinstance(value, tuple):
        return Kind.TUPLE

    if _is_table_like(value):
        return Kind.TABLE

    type_name = type(value).__name__
    module = type(value).__module__
    if type_name == "ndarray" and module.startswith("numpy"):
        return Kind.TENSOR

    return Kind.ANY


_TABLE_SAMPLE_SIZE = 16


def _is_table_like(value: Any) -> bool:
    """True for DataFrame-style objects from common Python data libraries.

    Restricted to the canonical "rectangular table" types of each
    library — sibling types like ``polars.Series`` or
    ``pandas.Index`` aren't tables and shouldn't be coerced into the
    TABLE renderer.
    """
    type_name = type(value).__name__
    module = type(value).__module__
    if module.startswith("pandas") and type_name == "DataFrame":
        return True
    if module.startswith("polars") and type_name in ("DataFrame", "LazyFrame"):
        return True
    if module.startswith("pyarrow") and type_name in ("Table", "RecordBatch"):
        return True
    # DuckDB's relation type lives in the C extension module ``_duckdb``,
    # so checking the (very distinctive) class name alone is enough.
    if type_name == "DuckDBPyRelation":
        return True
    return False


def serialize_value(
    value: Any,
    encoding: str = "json",
    view: VarView | None = None,
) -> tuple[Any, str | None]:
    """Serialize a value for the wire.

    Args:
        value: Python value to serialize.
        encoding: ``"json"`` or ``"arrow_ipc"``.
        view: Optional per-variable rendering hints (row limit, offset).
            Applied at the top-level table shape only. ``arrow_ipc``
            ignores ``view`` today; the IPC stream always carries the full
            table.

    Returns:
        (inline_value, blob_ref) — one of these will be non-None.
        If blob_ref is set, the value is too large for inline and should
        be served at that ref URL.
    """
    if encoding == "json":
        return _to_json(value, view=view), None
    if encoding == "arrow_ipc":
        return None, _to_arrow_ipc_ref(value)
    # Default: attempt JSON
    return _to_json(value, view=view), None


def _to_json(value: Any, view: VarView | None = None) -> Any:
    """Convert a Python value to a JSON-compatible form.

    ``view`` is only consulted for the *top-level* tabular shape. Nested
    tables (a dict of dataframes, a list of arrow tables) get serialized
    in full — applying a row offset/limit to a deeply nested structure
    has no obvious correct semantics, and the use case is "I subscribed
    to a table; render N rows of it." Subscribe to the nested handle
    directly if you need pagination of it.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}

    type_name = type(value).__name__
    module = type(value).__module__

    # Tabular values: honor ``view`` if present; otherwise return the full
    # frame. The default is unlimited — callers explicitly opt in to
    # truncation when they want a preview-shaped payload.
    if module.startswith("pandas") and type_name == "DataFrame":
        sliced = _apply_view_pandas(value, view)
        return sliced.to_dict(orient="records")
    if module.startswith("polars") and type_name == "DataFrame":
        return _apply_view_polars(value, view).to_dicts()
    if module.startswith("polars") and type_name == "LazyFrame":
        # ``slice`` on a LazyFrame is a query rewrite, so the bounds push
        # down — we never collect more than the requested window.
        return _apply_view_polars_lazy(value, view).collect().to_dicts()
    if module.startswith("pyarrow") and type_name in ("Table", "RecordBatch"):
        # ``slice`` on Arrow is zero-copy.
        return _apply_view_arrow(value, view).to_pylist()
    if type_name == "DuckDBPyRelation":
        # ``limit`` is a relational rewrite; the engine never produces
        # rows past the window. Pair each row tuple with the column
        # names so the client gets the JSON-records shape it expects.
        rel = _apply_view_duckdb(value, view)
        return [dict(zip(rel.columns, row)) for row in rel.fetchall()]

    if type_name == "ndarray" and module.startswith("numpy"):
        return value.tolist()

    try:
        return repr(value)
    except Exception:
        return f"<{type_name}>"


# ---------------------------------------------------------------------------
# Per-engine view application
#
# Each helper is the bare minimum that engine-specific slicing requires; the
# branches sit in the JSON path above. They're factored out so the call
# sites stay declarative and so a future "preview" hook can call into them
# directly without duplicating the dispatch logic.
# ---------------------------------------------------------------------------


def _apply_view_pandas(df: Any, view: VarView | None) -> Any:
    if view is None or (view.row_limit is None and view.row_offset == 0):
        return df
    start = view.row_offset
    stop = (
        start + view.row_limit if view.row_limit is not None else None
    )
    return df.iloc[start:stop]


def _apply_view_polars(df: Any, view: VarView | None) -> Any:
    if view is None or (view.row_limit is None and view.row_offset == 0):
        return df
    return df.slice(view.row_offset, view.row_limit)


def _apply_view_polars_lazy(lf: Any, view: VarView | None) -> Any:
    if view is None or (view.row_limit is None and view.row_offset == 0):
        return lf
    return lf.slice(view.row_offset, view.row_limit)


def _apply_view_arrow(value: Any, view: VarView | None) -> Any:
    if view is None or (view.row_limit is None and view.row_offset == 0):
        return value
    # ``Table.slice(offset, length)`` and ``RecordBatch.slice(offset, length)``
    # share the same signature; ``length=None`` clamps to end-of-table.
    if view.row_limit is None:
        return value.slice(view.row_offset)
    return value.slice(view.row_offset, view.row_limit)


def _apply_view_duckdb(rel: Any, view: VarView | None) -> Any:
    if view is None or (view.row_limit is None and view.row_offset == 0):
        return rel
    # ``limit(n, offset=k)`` is a relational rewrite that pushes the
    # window into the plan.
    if view.row_limit is None:
        # DuckDB's ``limit`` requires a count; use a very large sentinel
        # rather than collecting eagerly for "everything past offset".
        # Callers paginating without a stop should use successive offset
        # bumps instead.
        return rel.limit(2**63 - 1, offset=view.row_offset)
    return rel.limit(view.row_limit, offset=view.row_offset)


def _to_arrow_ipc_ref(value: Any) -> str:
    """Serialize a table to Arrow IPC and return a blob reference.

    For now this returns inline base64; in Phase 2 we'll use the
    virtual file system for large payloads.
    """
    import pyarrow as pa

    type_name = type(value).__name__
    module = type(value).__module__

    table: pa.Table
    if module.startswith("pandas") and type_name == "DataFrame":
        table = pa.Table.from_pandas(value)
    elif module.startswith("polars") and type_name == "DataFrame":
        table = value.to_arrow()
    elif module.startswith("polars") and type_name == "LazyFrame":
        table = value.collect().to_arrow()
    elif module.startswith("pyarrow") and type_name == "RecordBatch":
        table = pa.Table.from_batches([value])
    elif type_name == "DuckDBPyRelation":
        table = value.to_arrow_table()
    elif isinstance(value, pa.Table):
        table = value
    else:
        raise TypeError(
            f"Cannot serialize {type_name} as Arrow IPC. Expected a "
            "DataFrame (pandas/polars), pyarrow Table/RecordBatch, or "
            "DuckDBPyRelation."
        )

    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    buf = sink.getvalue()
    return (
        "data:application/vnd.apache.arrow.stream;base64,"
        + base64.b64encode(buf.to_pybytes()).decode("ascii")
    )
