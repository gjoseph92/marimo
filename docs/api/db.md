# Database

Reactive database handles. Reads through a handle register a dependency in
marimo's reactive graph, and writes — from this notebook or from another
process — cause dependent cells to re-run.

See [Reactive databases](../guides/reactive_databases.md) for a full walkthrough.

::: marimo.db.attach

::: marimo._db.ReactiveSQLite

::: marimo._db.ReactiveDatabase
