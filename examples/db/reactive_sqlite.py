# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import sqlite3
    import time

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Reactive SQLite

    `mo.db.attach` connects a SQLite database to marimo's reactive graph.
    Reads through the handle re-run automatically when the database
    changes — whether the change came from this notebook or from
    another process.
    """)
    return


@app.cell
def _():
    db_path = mo.notebook_dir() / "events.db"
    db = mo.db.attach(db_path)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    return db, db_path


@app.cell
def _():
    payload = mo.ui.text(placeholder="payload", label="Payload")
    add = mo.ui.run_button(label="Add event")
    mo.hstack([payload, add])
    return add, payload


@app.cell
def _(add, db, payload):
    if add.value and payload.value:
        db.execute(
            "INSERT INTO events(ts, payload) VALUES (?, ?)",
            (time.time(), payload.value),
        )
    return


@app.cell
def _(db):
    events = db.query(
        "SELECT id, ts, payload FROM events ORDER BY id DESC LIMIT 50"
    )
    events
    return (events,)


@app.cell
def _():
    clear = mo.ui.run_button(label="Clear all events", kind="danger")
    clear
    return (clear,)


@app.cell
def _(clear, db):
    if clear.value:
        db.execute("DELETE FROM events")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## External writers

    The button below opens its own `sqlite3` connection and inserts a row
    **without going through `db`**. The events table above still re-renders
    on the next watcher tick (about a second), because `mo.db.attach`
    polls the database file's mtime and bumps the reactive signal whenever
    another process commits.
    """)
    return


@app.cell
def _():
    external = mo.ui.run_button(label="Insert via raw sqlite3 (bypassing db)")
    external
    return (external,)


@app.cell
def _(db_path, external):
    if external.value:
        with sqlite3.connect(db_path) as _conn:
            _conn.execute(
                "INSERT INTO events(ts, payload) VALUES (?, ?)",
                (time.time(), "from another process"),
            )
            _conn.commit()
    return


@app.cell
def _(events):
    # another view of the `query` shown here for convenience
    events
    return


if __name__ == "__main__":
    app.run()
