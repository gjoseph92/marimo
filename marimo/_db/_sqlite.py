# Copyright 2026 Marimo. All rights reserved.
"""Reactive SQLite handle.

The handle is a `State[int]` whose value is a monotonically increasing version
stamp. It is bumped from two sources:

1. Writes performed through the handle (`execute`, `executemany`, `transaction`).
2. A background thread that polls the on-disk file's mtime and bumps the stamp
   when an external writer modifies the database.

Cells that statically reference the handle re-run whenever the stamp changes.
This is the same mechanism `mo.state` and `mo.watch.file` rely on.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from marimo import _loggers
from marimo._db._protocol import ResultFormat
from marimo._output.rich_help import mddoc
from marimo._runtime.context import (
    runtime_context_installed,
)
from marimo._runtime.state import State
from marimo._runtime.threads import Thread
from marimo._utils.platform import is_pyodide

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

LOGGER = _loggers.marimo_logger()

# How often the watcher thread polls the database file's mtime, in seconds.
# Matches the cadence used by `mo.watch.file`.
WATCHER_SLEEP_INTERVAL = 1.0

# Test-only override for the watcher poll interval. Production code should
# leave this as None.
_TEST_SLEEP_INTERVAL: float | None = None

JournalMode = Literal["wal", "delete", "truncate", "memory", "off", "persist"]


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _db_signature(path: Path) -> tuple[float, float, float]:
    """Return a tuple that changes whenever the SQLite database changes.

    SQLite in WAL mode writes to `<path>-wal` and `<path>-shm` until a
    checkpoint occurs. Hashing all three mtimes lets us catch external writes
    regardless of journal mode.
    """
    suffix = path.suffix
    return (
        _safe_mtime(path),
        _safe_mtime(path.with_suffix(suffix + "-wal")),
        _safe_mtime(path.with_suffix(suffix + "-shm")),
    )


@mddoc
class ReactiveSQLite(State[int]):
    """A reactive SQLite handle.

    Returned by [`mo.db.attach`][marimo.db.attach]. See the module-level
    documentation in [`marimo.db`][marimo._db] for usage.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        journal_mode: JournalMode = "wal",
        timeout: float = 5.0,
        memoize: bool = True,
        cache_size: int = 128,
        default_format: ResultFormat = "auto",
    ) -> None:
        super().__init__(0, allow_self_loops=False)

        self._path: Path | None
        self._is_memory = str(path) == ":memory:"
        self._path = None if self._is_memory else Path(path)
        self._timeout = timeout
        self._journal_mode = journal_mode
        self._memoize = memoize
        self._cache_size = cache_size
        self._default_format: ResultFormat = default_format

        self._init_lock = threading.Lock()
        self._initialized = False

        # In-memory SQLite databases die when their last connection closes.
        # Hold a single connection open for the lifetime of the handle and
        # reuse it for every operation; serialize access with a lock since
        # sqlite3 connections aren't safe to share across threads otherwise.
        self._memory_conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()

        self._cache: OrderedDict[
            tuple[int, str, tuple[Any, ...], ResultFormat], Any
        ] = OrderedDict()
        self._cache_lock = threading.Lock()

        self._should_exit = threading.Event()
        # In-memory databases have no file to watch; in Pyodide threads aren't
        # supported. In both cases local writes still bump the version.
        if (
            not self._is_memory
            and runtime_context_installed()
            and not is_pyodide()
        ):
            Thread(
                target=self._watch_loop,
                name=f"reactive-sqlite[{self._path}]",
                daemon=True,
            ).start()
        elif self._is_memory and runtime_context_installed():
            LOGGER.debug(
                "ReactiveSQLite(:memory:) skipping file watcher; only local "
                "writes will trigger reactive updates."
            )
        elif is_pyodide():
            LOGGER.warning(
                "ReactiveSQLite file watching is not supported in Pyodide; "
                "external writes will not trigger reactive updates."
            )

    def _watch_loop(self) -> None:
        assert self._path is not None
        sleep_interval = _TEST_SLEEP_INTERVAL or WATCHER_SLEEP_INTERVAL
        last = _db_signature(self._path)
        while not self._should_exit.is_set():
            time.sleep(sleep_interval)
            try:
                current = _db_signature(self._path)
            except Exception as exc:  # pragma: no cover - defensive
                sys.stderr.write(
                    f"Error watching sqlite database {self._path}: {exc}\n"
                )
                continue
            if current != last:
                last = current
                self._bump()

    def _bump(self) -> None:
        # Mutating self._value via _set_value also calls
        # ctx.register_state_update, which marks dependent cells stale.
        self._set_value(lambda v: v + 1)
        # Old cache entries are dead weight as soon as the version changes.
        with self._cache_lock:
            self._cache.clear()

    @property
    def path(self) -> Path | None:
        """The on-disk path being managed, or `None` for `:memory:` databases."""
        return self._path

    @property
    def version(self) -> int:
        """The current version stamp. Bumped on every write or external change."""
        return self._value

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._is_memory:
            with self._conn_lock:
                if self._memory_conn is None:
                    self._memory_conn = sqlite3.connect(
                        ":memory:",
                        timeout=self._timeout,
                        isolation_level=None,
                        check_same_thread=False,
                    )
                self._ensure_initialized(self._memory_conn)
                yield self._memory_conn
            return
        conn = sqlite3.connect(
            str(self._path),
            timeout=self._timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._ensure_initialized(conn)
            yield conn
        finally:
            conn.close()

    def _ensure_initialized(self, conn: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            if not self._is_memory:
                conn.execute(f"PRAGMA journal_mode={self._journal_mode}")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._initialized = True

    def query(
        self,
        sql: str,
        params: Iterable[Any] = (),
        *,
        as_: ResultFormat | None = None,
    ) -> Any:
        """Run a read query and return the result.

        Args:
            sql: The SQL statement to execute.
            params: Parameters to bind to the statement.
            as_: Result format. Defaults to the handle's `default_format`
                (which itself defaults to `"auto"`: polars if installed,
                then pandas, then records).

        Returns:
            A `polars.DataFrame`, `pandas.DataFrame`, list of dicts, or list
            of raw row tuples, depending on `as_`.

        Reads register a reactive dependency: dependent cells will re-run
        when this database is next written to.
        """
        fmt = as_ or self._default_format
        params_tuple = tuple(params)

        if self._memoize:
            key = (self._value, sql, params_tuple, fmt)
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return hit

        with self._connect() as conn:
            result = self._materialize(conn, sql, params_tuple, fmt)

        if self._memoize:
            with self._cache_lock:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return result

    def _materialize(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[Any, ...],
        fmt: ResultFormat,
    ) -> Any:
        resolved = _resolve_format(fmt)
        if resolved == "polars":
            import polars as pl

            return pl.read_database(
                sql, conn, execute_options={"parameters": params}
            )
        if resolved == "pandas":
            import pandas as pd

            return pd.read_sql_query(sql, conn, params=params)

        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        if resolved == "records":
            cols = [d[0] for d in cur.description or []]
            return [dict(zip(cols, row, strict=False)) for row in rows]
        return rows

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run a single write statement. Returns the affected rowcount.

        Bumps the reactive signal on success.
        """
        with self._connect() as conn:
            cur = conn.execute(sql, tuple(params))
            conn.commit()
            rowcount = cur.rowcount
        self._bump()
        return rowcount

    def executemany(
        self, sql: str, seq_of_params: Iterable[Iterable[Any]]
    ) -> None:
        """Run a write statement once per row in `seq_of_params`.

        Bumps the reactive signal once on completion, regardless of how many
        rows were inserted.
        """
        with self._connect() as conn:
            conn.executemany(sql, [tuple(p) for p in seq_of_params])
            conn.commit()
        self._bump()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Open a write transaction.

        Yields the underlying `sqlite3.Connection`. On clean exit the
        transaction is committed and the reactive signal is bumped exactly
        once. On exception the transaction is rolled back and no bump occurs.
        """
        committed = False
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            conn.commit()
            committed = True
        if committed:
            # Bump outside the connection scope so we don't hold the file open
            # while readers wake up.
            self._bump()

    def invalidate(self) -> None:
        """Force dependent cells to re-run."""
        self._bump()

    def cache_clear(self) -> None:
        """Drop all memoized query results."""
        with self._cache_lock:
            self._cache.clear()

    def close(self) -> None:
        """Stop the background watcher thread and release resources.

        After `close()`, external writes will no longer be detected
        reactively. For in-memory databases this also drops all data.
        """
        self._should_exit.set()
        with self._conn_lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._should_exit.set()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def __repr__(self) -> str:
        target = ":memory:" if self._is_memory else str(self._path)
        return f"ReactiveSQLite({target}, version={self._value})"


def _resolve_format(fmt: ResultFormat) -> ResultFormat:
    if fmt != "auto":
        return fmt
    try:
        import polars  # noqa: F401

        return "polars"
    except ImportError:
        pass
    try:
        import pandas  # noqa: F401

        return "pandas"
    except ImportError:
        pass
    return "records"


@mddoc
def attach(
    path: str | Path,
    *,
    journal_mode: JournalMode = "wal",
    timeout: float = 5.0,
    memoize: bool = True,
    cache_size: int = 128,
    default_format: ResultFormat = "auto",
) -> ReactiveSQLite:
    """Attach to a SQLite database with marimo-aware reactivity.

    Reads through the returned handle register a reactive dependency, and
    writes (through this handle or from another process) cause dependent
    cells to re-run.

    Examples:
        Open a database once, then read and write from separate cells.

        ```python
        db = mo.db.attach("app.db")
        ```

        ```python
        df = db.query("SELECT * FROM events ORDER BY ts DESC LIMIT 100")
        df
        ```

        ```python
        run = mo.ui.run_button(label="Insert event")
        run
        ```

        ```python
        if run.value:
            db.execute(
                "INSERT INTO events(ts, payload) VALUES (?, ?)",
                (time.time(), "hello"),
            )
        ```

        Both the reader cell and any other cell referencing `db` re-run
        immediately on `execute`, and on the next watcher tick when another
        process writes to the database file.

    Args:
        path: Path to the SQLite database, or `":memory:"`. Memory databases
            cannot be watched for external changes.
        journal_mode: Initial `PRAGMA journal_mode`. Defaults to `"wal"`,
            which is the right choice for concurrent readers and writers.
        timeout: SQLite busy timeout in seconds, passed to `sqlite3.connect`.
        memoize: When `True`, identical queries within the same version are
            served from an in-memory LRU. The cache is automatically cleared
            on every write or external change.
        cache_size: Maximum number of memoized query results.
        default_format: Default value for the `as_` argument of
            [`ReactiveSQLite.query`][marimo._db.ReactiveSQLite.query].

    Returns:
        A [`ReactiveSQLite`][marimo._db.ReactiveSQLite] handle.
    """
    return ReactiveSQLite(
        path,
        journal_mode=journal_mode,
        timeout=timeout,
        memoize=memoize,
        cache_size=cache_size,
        default_format=default_format,
    )
