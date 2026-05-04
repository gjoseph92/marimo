# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING

import pytest
from inline_snapshot import snapshot

import marimo as mo
from marimo._db._sqlite import ReactiveSQLite, _resolve_format
from marimo._dependencies.dependencies import DependencyManager

if TYPE_CHECKING:
    from pathlib import Path

    from marimo._runtime.runtime import Kernel
    from tests.conftest import ExecReqProvider


HAS_POLARS = DependencyManager.polars.has()
HAS_PANDAS = DependencyManager.pandas.has()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "app.db"
    with sqlite3.connect(p) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO t(name) VALUES ('alice'), ('bob')")
        conn.commit()
    return p


def test_records_format(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records")
    try:
        assert db.query("SELECT * FROM t ORDER BY id") == snapshot(
            [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        )
    finally:
        db.close()


def test_execute_bumps_version_and_invalidates_cache(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records")
    try:
        v0 = db.version
        first = db.query("SELECT count(*) AS n FROM t")
        # Cached on second call; same version, same key.
        assert db.query("SELECT count(*) AS n FROM t") is first

        rowcount = db.execute("INSERT INTO t(name) VALUES (?)", ("carol",))

        assert (rowcount, db.version - v0) == snapshot((1, 1))
        # Version bumped, cache cleared, next read sees the write.
        assert db.query("SELECT count(*) AS n FROM t") == snapshot([{"n": 3}])
    finally:
        db.close()


def test_transaction_commits_and_bumps_once(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records", memoize=False)
    try:
        v0 = db.version
        with db.transaction() as conn:
            conn.execute("INSERT INTO t(name) VALUES ('c')")
            conn.execute("INSERT INTO t(name) VALUES ('d')")

        assert (
            db.version - v0,
            len(db.query("SELECT id FROM t")),
        ) == snapshot((1, 4))
    finally:
        db.close()


def test_transaction_rolls_back_and_does_not_bump(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records", memoize=False)

    def _insert_then_raise() -> None:
        with db.transaction() as conn:
            conn.execute("INSERT INTO t(name) VALUES ('zz')")
            raise RuntimeError("boom")

    try:
        v0 = db.version
        with pytest.raises(RuntimeError, match="boom"):
            _insert_then_raise()
        assert (
            db.version - v0,
            len(db.query("SELECT id FROM t")),
        ) == snapshot((0, 2))
    finally:
        db.close()


def test_executemany_bumps_once(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records", memoize=False)
    try:
        v0 = db.version
        db.executemany(
            "INSERT INTO t(name) VALUES (?)",
            [("c",), ("d",), ("e",)],
        )
        assert (
            db.version - v0,
            len(db.query("SELECT id FROM t")),
        ) == snapshot((1, 5))
    finally:
        db.close()


def test_invalidate_bumps_and_clears_cache(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records")
    try:
        first = db.query("SELECT 1 AS x")
        v0 = db.version
        db.invalidate()
        # Different object after invalidate (cache cleared, version bumped).
        assert db.version - v0 == 1
        assert db.query("SELECT 1 AS x") is not first
    finally:
        db.close()


def test_memoize_off_recomputes(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records", memoize=False)
    try:
        a = db.query("SELECT 1 AS x")
        b = db.query("SELECT 1 AS x")
        assert a == b
        assert a is not b
        assert len(db._cache) == 0
    finally:
        db.close()


def test_cache_lru_eviction(db_path: Path) -> None:
    db = ReactiveSQLite(db_path, default_format="records", cache_size=2)
    try:
        db.query("SELECT 1 AS x")
        db.query("SELECT 2 AS x")
        db.query("SELECT 3 AS x")
        assert len(db._cache) == snapshot(2)
    finally:
        db.close()


def test_memory_database_no_watcher() -> None:
    db = ReactiveSQLite(":memory:", default_format="records")
    try:
        # `path` is None for in-memory and writes still bump the version.
        v0 = db.version
        db.execute("CREATE TABLE t(x INT)")
        db.execute("INSERT INTO t VALUES (?)", (42,))
        assert (
            db.path,
            db.version - v0,
            db.query("SELECT * FROM t"),
        ) == snapshot((None, 2, [{"x": 42}]))
    finally:
        db.close()


def test_resolve_format_auto_prefers_polars() -> None:
    fmt = _resolve_format("auto")
    if HAS_POLARS:
        assert fmt == "polars"
    elif HAS_PANDAS:
        assert fmt == "pandas"
    else:
        assert fmt == "records"


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_format(db_path: Path) -> None:
    import polars as pl

    db = ReactiveSQLite(db_path, default_format="polars")
    try:
        df = db.query("SELECT id, name FROM t ORDER BY id")
        assert isinstance(df, pl.DataFrame)
        assert df.to_dicts() == snapshot(
            [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        )
    finally:
        db.close()


def test_namespace_export() -> None:
    # Public API surface check; matters because this is the only thing in
    # `mo.db`.
    assert mo.db.attach is not None
    assert mo.db.ReactiveSQLite is ReactiveSQLite


@pytest.mark.xfail(
    True,
    reason="Timing-based watcher test; flaky in CI like sibling watch tests",
    strict=False,
)
async def test_external_write_triggers_reactivity_in_kernel(
    execution_kernel: Kernel,
    exec_req: ExecReqProvider,
    tmp_path: Path,
) -> None:
    """End-to-end: an out-of-band write to the SQLite file invalidates a
    cell that reads through the handle."""
    db_file = tmp_path / "ext.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE t(v INT)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()

    await execution_kernel.run(
        [
            exec_req.get(
                "import marimo as mo\n"
                "from marimo._db import _sqlite as _s\n"
                "_s._TEST_SLEEP_INTERVAL = 0.02"
            ),
            exec_req.get(
                f'db = mo.db.attach(r"{db_file.as_posix()}", '
                f'memoize=False, default_format="records")'
            ),
            exec_req.get('rows = db.query("SELECT v FROM t ORDER BY v")'),
        ]
    )
    assert execution_kernel.globals["rows"] == [{"v": 1}]

    with sqlite3.connect(db_file) as c:
        c.execute("INSERT INTO t VALUES (2)")
        c.commit()

    for _ in range(50):
        await asyncio.sleep(0.05)
        if execution_kernel.globals.get("rows") == [{"v": 1}, {"v": 2}]:
            break
    assert execution_kernel.globals["rows"] == [{"v": 1}, {"v": 2}]
