# Copyright 2026 Marimo. All rights reserved.
"""Static-only schema tests.

Live ``mo.api.input`` schema generation runs on the kernel and is exercised
end-to-end via the dataflow API integration tests; this module covers the
graph-only fallback used by tooling.
"""

from __future__ import annotations

import marimo
from marimo._ast.app import InternalApp
from marimo._dataflow.schema import (
    compute_dataflow_schema,
    compute_dataflow_schema_from_globals,
)


class TestStaticSchema:
    def test_detects_free_vars_as_inputs(self) -> None:
        app = marimo.App()

        @app.cell
        def _(x, y):
            result = x + y
            return (result,)

        schema = compute_dataflow_schema(InternalApp(app))
        assert {i.name for i in schema.inputs} == {"x", "y"}
        assert {o.name for o in schema.outputs} == {"result"}

    def test_excludes_builtins(self) -> None:
        app = marimo.App()

        @app.cell
        def _(x):
            n = len(x)
            return (n,)

        schema = compute_dataflow_schema(InternalApp(app))
        assert "len" not in {i.name for i in schema.inputs}

    def test_schema_id_changes_with_structure(self) -> None:
        a, b = marimo.App(), marimo.App()

        @a.cell
        def _(x, y):
            r = x + y
            return (r,)

        @b.cell
        def _(z):
            r = z + 1
            return (r,)

        s1 = compute_dataflow_schema(InternalApp(a))
        s2 = compute_dataflow_schema(InternalApp(b))
        assert s1.schema_id != s2.schema_id

    def test_explicit_schema_id_is_passthrough(self) -> None:
        app = marimo.App()

        @app.cell
        def _(x):
            r = x + 1
            return (r,)

        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={},
            schema_id="static-test",
        )
        assert schema.schema_id == "static-test"


class TestKernelDerivedSchema:
    """Kernel-derived schema tests that exercise live globals introspection.

    These mirror what happens after the kernel instantiates a notebook —
    inputs come from ``mo.api.input``-tagged ``UIElement`` instances in
    globals; outputs come from graph defs filtered against globals values.
    """

    def test_run_button_input_gets_run_button_ui_constraint(self) -> None:
        import marimo as mo

        app = marimo.App()

        @app.cell
        def _():
            send = mo.api.input(ui=mo.ui.run_button(label="Send"))
            return (send,)

        send = mo.api.input(ui=mo.ui.run_button(label="Send"))
        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={"send": send},
        )
        [send_input] = [i for i in schema.inputs if i.name == "send"]
        assert send_input.constraints == {"ui": "run_button"}

    def test_explicit_ui_introspects_slider_bounds(self) -> None:
        """``ui=mo.ui.range_slider(...)`` surfaces start/stop/step constraints.

        The user never spelled the bounds in ``mo.api.input``, so the schema
        must walk the element to discover them — otherwise downstream agents
        only see ``{"ui": "range_slider"}`` and have to guess valid values.
        """
        import marimo as mo

        app = marimo.App()

        @app.cell
        def _():
            month_pair = mo.api.input(
                ui=mo.ui.range_slider(start=0, stop=23, step=1, value=[0, 11])
            )
            slider_pct = mo.api.input(ui=mo.ui.slider(start=0, stop=100))
            return month_pair, slider_pct

        month_pair = mo.api.input(
            ui=mo.ui.range_slider(start=0, stop=23, step=1, value=[0, 11])
        )
        slider_pct = mo.api.input(ui=mo.ui.slider(start=0, stop=100))
        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={"month_pair": month_pair, "slider_pct": slider_pct},
        )
        by_name = {i.name: i for i in schema.inputs}
        assert by_name["month_pair"].constraints == {
            "min": 0,
            "max": 23,
            "step": 1,
            "ui": "range_slider",
        }
        assert by_name["slider_pct"].constraints == {
            "min": 0,
            "max": 100,
            "ui": "slider",
        }

    def test_explicit_ui_dropdown_surfaces_options(self) -> None:
        import marimo as mo

        app = marimo.App()

        @app.cell
        def _():
            kind = mo.api.input(ui=mo.ui.dropdown(options=["a", "b", "c"]))
            return (kind,)

        kind = mo.api.input(ui=mo.ui.dropdown(options=["a", "b", "c"]))
        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={"kind": kind},
        )
        [kind_input] = [i for i in schema.inputs if i.name == "kind"]
        assert kind_input.constraints == {
            "options": ["a", "b", "c"],
            "ui": "dropdown",
        }

    def test_typing_helpers_filtered_from_outputs(self) -> None:
        """``from typing import Annotated`` shouldn't leak into outputs."""
        from typing import Annotated, Optional

        app = marimo.App()

        @app.cell
        def _():
            stats = {"count": 0}
            return (stats,)

        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={
                "Annotated": Annotated,
                "Optional": Optional,
                "stats": {"count": 0},
            },
        )
        names = {o.name for o in schema.outputs}
        assert "Annotated" not in names
        assert "Optional" not in names

    def test_output_annotation_attaches_description(self) -> None:
        import typing

        import marimo as mo

        app = marimo.App()

        @app.cell
        def _():
            stats = {"count": 0}
            return (stats,)

        schema = compute_dataflow_schema_from_globals(
            graph=InternalApp(app).graph,
            globals_={
                "stats": {"count": 0},
                "__annotations__": {
                    "stats": typing.Annotated[
                        dict, mo.api.output(description="Top-line stats")
                    ],
                },
            },
        )
        [stats_out] = [o for o in schema.outputs if o.name == "stats"]
        assert stats_out.description == "Top-line stats"
