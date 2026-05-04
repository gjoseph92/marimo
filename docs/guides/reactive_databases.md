# Reactive databases

`mo.db.attach` connects a SQLite database to marimo's reactive graph: reads
register a dependency, and writes — whether from your notebook or from another
process — cause dependent cells to re-run.

```python
import marimo as mo

db = mo.db.attach("app.db")
```

```python
df = db.query("SELECT * FROM events ORDER BY ts DESC LIMIT 100")
df
```

```python
run = mo.ui.run_button(label="Add event")
run
```

```python
import time

if run.value:
    db.execute(
        "INSERT INTO events(ts, payload) VALUES (?, ?)",
        (time.time(), "hello"),
    )
```

The reader cell re-runs as soon as the writer cell calls `db.execute(...)`.
If a separate process writes to `app.db`, the reader cell re-runs the next
time marimo's file watcher polls the database (about once a second).

## How it works

A handle returned by `mo.db.attach` is a [reactive
state][marimo.state] whose value is a version stamp. The stamp is bumped
whenever:

- you call `db.execute`, `db.executemany`, or commit a `db.transaction()`,
- a background watcher detects that the database file's mtime changed,
- you call `db.invalidate()` explicitly.

Cells that reference the handle (via `db.query(...)`) re-run on every bump.
This is the same machinery used by [`mo.state`][marimo.state] and
[`mo.watch.file`][marimo.watch.file], so the reactivity composes cleanly with
both.

## Memoization

Identical queries within the same version are served from an in-memory LRU,
which is automatically cleared on every write or external change. Disable
with `memoize=False` for very large result sets, or call `db.cache_clear()`
to flush manually.

```python
db = mo.db.attach("app.db", memoize=False)
```

## Result formats

`query()` returns a `polars.DataFrame` by default if polars is installed,
falling back to pandas, then to a list of dicts. Pin the format explicitly
when you care:

```python
db.query("SELECT * FROM t", as_="pandas")
db.query("SELECT * FROM t", as_="records")  # list[dict[str, Any]]
db.query("SELECT * FROM t", as_="raw")      # list[tuple]
```

Set the default per-handle with `default_format=`.

## Transactions

`transaction()` yields the underlying `sqlite3.Connection`. The transaction
commits on clean exit and bumps the reactive signal exactly once, regardless
of how many statements ran. On exception it rolls back and does not bump.

```python
with db.transaction() as conn:
    conn.execute("UPDATE balances SET v = v - ? WHERE id = ?", (10, src))
    conn.execute("UPDATE balances SET v = v + ? WHERE id = ?", (10, dst))
```

## Concurrency caveats

- `journal_mode="wal"` (the default) is the right setting for concurrent
  readers and writers. The watcher tracks both `app.db` and `app.db-wal` so
  WAL writes from other processes are detected immediately.
- Connections are short-lived: each call opens, runs, and closes its own
  connection. Don't try to share connections across cells.
- The watcher polls at one-second cadence by default. If you need faster
  reaction times to local writes, those happen synchronously through
  `execute()` and don't depend on the watcher at all.

## When _not_ to use this

If you have many concurrent writers and need sub-second freshness across
processes, SQLite is the wrong substrate. Use a database that exposes change
notifications (Postgres `LISTEN/NOTIFY`, a message queue, etc.) and feed
events into a [`mo.state`][marimo.state] from an `mo.Thread`.
