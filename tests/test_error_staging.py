"""Test documenting *when* a typo'd column name fails in different embeddings.

The key insight is that "staging" — when an error manifests — depends on
whether the implementation is:

  eager (immediate):  filter() walks the column dict right now, so a missing
                      column raises KeyError the moment .filter() is called.

  lazy / plan-building (deferred to collect):  filter() merely records the
                      predicate and column name in a plan node or a closure;
                      the column dict is not consulted until .collect() runs.
                      This applies to fluent_pushdown, lazy_pull, codegen, and
                      the functional / deep / pipe_rows families.

  arrow (immediate but different exception):  filter() receives a pc.Expression
                      object (not a Python lambda), which pyarrow validates
                      against the live Arrow schema *immediately* at filter()
                      call time.  The error is pa.lib.ArrowInvalid, not
                      KeyError.

All claims below are verified empirically; see the inline comments.
"""

from typing import Any

import pyarrow as pa
import pytest

_COLS: dict[str, list[Any]] = {"a": [1, 2, 3], "b": ["x", "y", "z"]}


# ---------------------------------------------------------------------------
# eager: KeyError at .filter() call time
# ---------------------------------------------------------------------------


def test_eager_filter_raises_immediately() -> None:
    """eager.DataFrame.filter() iterates over the column dict right now to build
    the mask, so a missing column name raises KeyError before the call returns."""
    from micro_dataframes import eager

    df = eager.DataFrame(_COLS)
    with pytest.raises(KeyError):
        # The error is raised here, at the .filter() call, not later.
        df.filter("nonexistent", lambda v: v == 1)


# ---------------------------------------------------------------------------
# fluent_pushdown: plan is built at .filter(), error deferred to .collect()
# ---------------------------------------------------------------------------


def test_fluent_pushdown_filter_deferred_to_collect() -> None:
    """fluent_pushdown.filter() wraps the column name and predicate in a Filter
    plan node — no column-dict lookup occurs yet.  The error only surfaces when
    .collect() walks the plan and calls execute(), which iterates the rows."""
    from micro_dataframes import fluent_pushdown

    df = fluent_pushdown.DataFrame(_COLS)
    # No error here — a plan node is built, not executed.
    ir = df.filter("nonexistent", lambda v: v == 1)
    # The error is raised here, at collect() time.
    with pytest.raises(KeyError):
        ir.collect()


# ---------------------------------------------------------------------------
# lazy_pull: closure built at .filter(), error deferred to .collect()
# ---------------------------------------------------------------------------


def test_lazy_pull_filter_deferred_to_collect() -> None:
    """lazy_pull.filter() builds a generator closure over the bad column name;
    the closure is not iterated until .collect() pulls rows through it."""
    from micro_dataframes import lazy_pull

    df = lazy_pull.DataFrame(_COLS)
    # No error here — a closure is constructed, not called.
    ir = df.filter("nonexistent", lambda v: v == 1)
    # The error is raised here when the generator is iterated inside collect().
    with pytest.raises(KeyError):
        ir.collect()


# ---------------------------------------------------------------------------
# codegen: plan node at .filter(), KeyError raised inside generated kernel
# ---------------------------------------------------------------------------


def test_codegen_filter_deferred_to_collect() -> None:
    """codegen.filter() records the column name in a Filter plan node.  At
    .collect() time the plan is compiled and executed.  The codegen path tracks
    column names in col_vars; if the column is not in the source schema the
    KeyError surfaces inside the generated kernel at runtime."""
    from micro_dataframes import codegen

    df = codegen.DataFrame(_COLS)
    # No error here.
    ir = df.filter("nonexistent", lambda v: v == 1)
    # The error surfaces inside the compiled kernel during collect().
    with pytest.raises(KeyError):
        ir.collect()


# ---------------------------------------------------------------------------
# arrow: pa.lib.ArrowInvalid raised *at* .filter() call time
# ---------------------------------------------------------------------------


def test_arrow_filter_raises_immediately_with_arrow_invalid() -> None:
    """arrow.DataFrame.filter() calls pa.Table.filter() with a pc.Expression
    built from arrow.col('nonexistent').  pyarrow validates the expression
    against the live Arrow schema at the moment filter() is invoked, so the
    error surfaces immediately — earlier than the lazy Python implementations
    but for a different reason (schema validation) than the eager one
    (column-dict iteration).  The exception is pa.lib.ArrowInvalid."""
    from micro_dataframes import arrow as arrow_mod

    df = arrow_mod.DataFrame(_COLS)
    # pa.lib.ArrowInvalid is raised here, at the .filter() call, not at collect().
    with pytest.raises(pa.lib.ArrowInvalid):
        df.filter(arrow_mod.col("nonexistent") == 1)
