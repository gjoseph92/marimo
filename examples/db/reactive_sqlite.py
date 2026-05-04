# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo

    return mo, time


@app.cell
def _(mo):
    mo.md(
        """
        # Reactive SQLite

        `mo.db.attach` connects a SQLite database to marimo's reactive graph.
        Reads through the handle re-run automatically when the database
        changes — whether the change came from this notebook or from
        another process.
        """
    )
    return


@app.cell
def _(mo):
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
    return (db,)


@app.cell
def _(mo):
    payload = mo.ui.text(placeholder="payload", label="Payload")
    add = mo.ui.run_button(label="Add event")
    mo.hstack([payload, add])
    return add, payload


@app.cell
def _(add, db, payload, time):
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
def _(mo):
    clear = mo.ui.run_button(label="Clear all events", kind="danger")
    clear
    return (clear,)


@app.cell
def _(clear, db):
    if clear.value:
        db.execute("DELETE FROM events")
    return


if __name__ == "__main__":
    app.run()
