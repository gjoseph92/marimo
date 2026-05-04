# Copyright 2026 Marimo. All rights reserved.
"""Reactive database connectors for marimo notebooks.

This package exposes [`mo.db.attach`][marimo.db.attach], a thin wrapper around
external databases that participates in marimo's reactive graph: cells that
read from a handle re-run automatically when the underlying database changes,
whether the change came from another cell, another process, or another machine.

Today only SQLite is supported; the [`ReactiveDatabase`][marimo._db.ReactiveDatabase]
protocol exists so additional engines can be added without breaking the API.
"""

from marimo._db._protocol import ReactiveDatabase, ResultFormat
from marimo._db._sqlite import ReactiveSQLite, attach

__all__ = [
    "ReactiveDatabase",
    "ReactiveSQLite",
    "ResultFormat",
    "attach",
]
