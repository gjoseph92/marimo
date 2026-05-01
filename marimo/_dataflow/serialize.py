# Copyright 2026 Marimo. All rights reserved.
"""Serialize Python values for the dataflow wire protocol."""

from __future__ import annotations

import base64
from typing import Any

from marimo._dataflow.protocol import Kind


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

# Row cap for table-shaped values serialized as JSON. The JSON channel
# is the streaming preview path — sending more rows than the client
# can display just wastes wire bytes and forces the producer to
# materialize the full table on every update. Use the ``arrow_ipc``
# encoding instead when you need the full payload.
JSON_TABLE_ROW_LIMIT = 100


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
) -> tuple[Any, str | None]:
    """Serialize a value for the wire.

    Returns:
        (inline_value, blob_ref) — one of these will be non-None.
        If blob_ref is set, the value is too large for inline and should
        be served at that ref URL.
    """
    if encoding == "json":
        return _to_json(value), None
    if encoding == "arrow_ipc":
        return None, _to_arrow_ipc_ref(value)
    # Default: attempt JSON
    return _to_json(value), None


def _to_json(value: Any) -> Any:
    """Convert a Python value to a JSON-compatible form."""
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

    # All table-shaped values get capped at JSON_TABLE_ROW_LIMIT — the
    # JSON channel is the preview path and full materialization on
    # every update is the surprise we explicitly don't want here.
    n = JSON_TABLE_ROW_LIMIT
    if module.startswith("pandas") and type_name == "DataFrame":
        return value.head(n).to_dict(orient="records")
    if module.startswith("polars") and type_name == "DataFrame":
        return value.head(n).to_dicts()
    if module.startswith("polars") and type_name == "LazyFrame":
        # ``head`` on a LazyFrame is a query rewrite, so the limit
        # pushes down — we never collect the full plan.
        return value.head(n).collect().to_dicts()
    if module.startswith("pyarrow") and type_name in ("Table", "RecordBatch"):
        # ``slice`` on Arrow is zero-copy and bounded by ``n``.
        return value.slice(0, n).to_pylist()
    if type_name == "DuckDBPyRelation":
        # ``limit`` is a relational rewrite; the engine never produces
        # the rows past ``n``. Pair each row tuple with the column
        # names so the client gets the JSON-records shape it expects.
        limited = value.limit(n)
        return [dict(zip(limited.columns, row)) for row in limited.fetchall()]

    if type_name == "ndarray" and module.startswith("numpy"):
        return value.tolist()

    try:
        return repr(value)
    except Exception:
        return f"<{type_name}>"


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
