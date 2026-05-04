# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterable
    from contextlib import AbstractContextManager


ResultFormat = Literal["auto", "polars", "pandas", "records", "raw"]
"""Return shape for [`ReactiveDatabase.query`][marimo._db.ReactiveDatabase.query].

- `"polars"`: a `polars.DataFrame`.
- `"pandas"`: a `pandas.DataFrame`.
- `"records"`: a list of `dict[column, value]` rows.
- `"raw"`: the raw rows produced by the underlying driver.
- `"auto"`: prefer polars, fall back to pandas, then records.
"""


@runtime_checkable
class ReactiveDatabase(Protocol):
    """A reactive database handle.

    Reads (`query`) register a dependency on the handle, and writes (`execute`,
    `executemany`, `transaction`) bump that dependency so cells reading from
    the handle re-run automatically. External writers are detected via
    file/connection watching where supported by the underlying engine.

    The handle itself must be a [`marimo.state`][marimo.state]-like object so
    that referring to it from a cell creates a reactive dependency. The
    bundled [`ReactiveSQLite`][marimo._db.ReactiveSQLite] implementation
    inherits from `marimo._runtime.state.State` to satisfy this contract.
    """

    def query(
        self,
        sql: str,
        params: Iterable[Any] = (),
        *,
        as_: ResultFormat | None = None,
    ) -> Any:
        """Run a read query and return the result in the requested format."""
        ...

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run a single write statement. Returns affected rowcount.

        Bumps the reactive signal on success so dependent cells re-run.
        """
        ...

    def executemany(
        self, sql: str, seq_of_params: Iterable[Iterable[Any]]
    ) -> None:
        """Run a write statement once per row in `seq_of_params`."""
        ...

    def transaction(self) -> AbstractContextManager[Any]:
        """Open a write transaction; commits on exit and bumps once.

        The yielded value is the underlying connection. The transaction
        rolls back on exception and does not bump the reactive signal.
        """
        ...

    def invalidate(self) -> None:
        """Force dependent cells to re-run.

        Use this after performing writes through a path that bypasses this
        handle (for example, an offline migration) and you want readers in
        the current notebook to refresh immediately.
        """
        ...

    def cache_clear(self) -> None:
        """Clear any memoized query results."""
        ...

    def close(self) -> None:
        """Stop background watchers and release resources."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: Any) -> None: ...


__all__ = ["ReactiveDatabase", "ResultFormat"]
