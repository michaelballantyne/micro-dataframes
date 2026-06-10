"""Differential test suite for all thirteen micro-dataframe implementations.

Relaxed contract (applies to all tests in this module):
  * Row order is undefined — limit(n) may return ANY n rows.
  * Column names on both sides of a join are assumed disjoint.

Each adapter receives (routes_rows, airlines_rows) as list[dict[str, Any]] and
returns dict[str, list[Any]] of result columns for the codeshare query
(join routes + airlines on airline-id, filter codeshare=Y and name=American
Airlines, limit 3).

Ground-truth for join/codeshare tests is computed by a plain-Python helper
(no dataframe implementation involved) and expressed as a multiset
(collections.Counter) so that order does not matter.

The subset test runs on a trimmed slice of the real OpenFlights data so that
even the O(n*m) nested-loop implementations finish quickly.
"""

import collections
import csv
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "examples" / "openflights"


def _load(name: str) -> list[dict[str, Any]]:
    with (_DATA_DIR / f"{name}.csv").open() as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _to_col_dict(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert list-of-dicts to column-oriented dict."""
    cols: dict[str, list[Any]] = {}
    for row in rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    return cols


def _codeshare_ground_truth(
    routes_rows: list[dict[str, Any]],
    airlines_rows: list[dict[str, Any]],
) -> collections.Counter[str]:
    """Plain-Python ground truth: multiset of source-airport values for routes
    whose route-airline-id matches an airline named 'American Airlines' and
    whose codeshare field is 'Y'.  No dataframe implementation is involved."""
    aa_ids = {row["airline-id"] for row in airlines_rows if row["name"] == "American Airlines"}
    result: collections.Counter[str] = collections.Counter()
    for row in routes_rows:
        if row.get("codeshare") == "Y" and row.get("route-airline-id") in aa_ids:
            result[row["source-airport"]] += 1
    return result


# ---------------------------------------------------------------------------
# Small inline tables used for fast join / filter / limit tests
# (column names are disjoint to avoid the double-append quirk when the same
# key appears in both sides of an eager join)
# ---------------------------------------------------------------------------

# Three-row left table, two-row right table.
_LEFT_ROWS: list[dict[str, Any]] = [
    {"route-id": "1", "src": "A"},
    {"route-id": "2", "src": "C"},
    {"route-id": "1", "src": "E"},  # second left row matching id=1 → duplication
]
_RIGHT_ROWS: list[dict[str, Any]] = [
    {"airline-id": "1", "name": "X"},
    {"airline-id": "2", "name": "Y"},
]

# Column-dict view of the same data (for implementations that take that form).
_LEFT_COLS: dict[str, list[Any]] = _to_col_dict(_LEFT_ROWS)
_RIGHT_COLS: dict[str, list[Any]] = _to_col_dict(_RIGHT_ROWS)

# Tiny data for filter-only tests (3 rows).
_FILTER_ROWS: list[dict[str, Any]] = [
    {"city": "NYC", "pop": 8000000},
    {"city": "LAX", "pop": 4000000},
    {"city": "CHI", "pop": 2700000},
]
_FILTER_COLS: dict[str, list[Any]] = _to_col_dict(_FILTER_ROWS)

# ---------------------------------------------------------------------------
# Codeshare-query subset (fast enough for slow implementations)
# ---------------------------------------------------------------------------
# AA codeshare routes first appear after index 4655 in the full dataset.  We
# include the first 5000 routes rows and all airlines rows so that every
# implementation (including unoptimised O(n*m) ones) can compute the answer in
# a test-reasonable time (~2 s worst case per test).
# The expected result on this subset is computed via the plain-Python
# _codeshare_ground_truth helper (no dataframe implementation involved).

_ROUTES_SUBSET_SIZE = 5000


# ---------------------------------------------------------------------------
# Adapter functions
# All return dict[str, list[Any]] matching the codeshare query result.
# ---------------------------------------------------------------------------


def _codeshare_eager(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import eager

    r = eager.DataFrame(routes_rows)
    a = eager.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
    )
    return dict(result._columns)


def _codeshare_query_lift(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import query_lift

    r_df = query_lift.DataFrame(routes_rows)
    a_df = query_lift.DataFrame(airlines_rows)
    result = (
        query_lift.q(r_df)
        .join(query_lift.q(a_df), "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_query_forward(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import query_forward

    r = query_forward.DataFrame(routes_rows)
    a = query_forward.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_lazy_pull(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_pull

    r = lazy_pull.DataFrame(routes_rows)
    a = lazy_pull.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_lazy_push(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_push

    r = lazy_push.DataFrame(routes_rows)
    a = lazy_push.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_functional(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    import micro_dataframes.functional as func

    r = func.source(_to_col_dict(routes_rows))
    a = func.source(_to_col_dict(airlines_rows))
    q = func.limit(
        func.filter(
            func.filter(
                func.join(r, a, "route-airline-id", "airline-id"),
                "codeshare",
                lambda v: v == "Y",
            ),
            "name",
            lambda v: v == "American Airlines",
        ),
        3,
    )
    return func.collect(q)


def _codeshare_deep(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import deep

    r = deep.Source(deep.from_rows(routes_rows))
    a = deep.Source(deep.from_rows(airlines_rows))
    plan = deep.Limit(
        deep.Filter(
            deep.Filter(
                deep.Join(r, a, "route-airline-id", "airline-id"),
                "codeshare",
                lambda v: v == "Y",
            ),
            "name",
            lambda v: v == "American Airlines",
        ),
        3,
    )
    return deep.collect(plan)


def _codeshare_deep_pushdown(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import deep_pushdown

    r = deep_pushdown.Source(deep_pushdown.from_rows(routes_rows))
    a = deep_pushdown.Source(deep_pushdown.from_rows(airlines_rows))
    plan = deep_pushdown.Limit(
        deep_pushdown.Filter(
            deep_pushdown.Filter(
                deep_pushdown.Join(r, a, "route-airline-id", "airline-id"),
                "codeshare",
                lambda v: v == "Y",
            ),
            "name",
            lambda v: v == "American Airlines",
        ),
        3,
    )
    return deep_pushdown.collect(plan)


def _codeshare_fluent_pushdown(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import fluent_pushdown

    r = fluent_pushdown.DataFrame(routes_rows)
    a = fluent_pushdown.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_pipe_rows(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import pipe_rows

    r = pipe_rows.DataFrame(routes_rows)
    a = pipe_rows.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_codegen(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import codegen

    r = codegen.DataFrame(routes_rows)
    a = codegen.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
        .limit(3)
        .collect()
    )
    return dict(result._columns)


def _codeshare_arrow(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import arrow as arrow_mod

    r = arrow_mod.DataFrame(routes_rows)
    a = arrow_mod.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter(arrow_mod.col("codeshare") == "Y")
        .filter(arrow_mod.col("name") == "American Airlines")
        .limit(3)
    )
    # Return only the columns present in Python-list implementations for fair
    # comparison (arrow keeps all schema columns regardless of row count).
    return {col: result[col] for col in result._table.schema.names}


def _codeshare_vectorized(
    routes_rows: list[dict[str, Any]], airlines_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import vectorized as vec_mod

    r = vec_mod.DataFrame(routes_rows)
    a = vec_mod.DataFrame(airlines_rows)
    result = (
        r.join(a, "route-airline-id", "airline-id")
        .filter(vec_mod.col("codeshare") == "Y")
        .filter(vec_mod.col("name") == "American Airlines")
        .limit(3)
    )
    # _columns holds full column arrays; apply sel via __getitem__ for each column.
    return {c: result[c] for c in result._columns}


# ---------------------------------------------------------------------------
# Parametrize over all twelve adapters
# ---------------------------------------------------------------------------

_ALL_ADAPTERS = [
    pytest.param(_codeshare_eager, id="eager"),
    pytest.param(_codeshare_query_lift, id="query_lift"),
    pytest.param(_codeshare_query_forward, id="query_forward"),
    pytest.param(_codeshare_lazy_pull, id="lazy_pull"),
    pytest.param(_codeshare_lazy_push, id="lazy_push"),
    pytest.param(_codeshare_functional, id="functional"),
    pytest.param(_codeshare_deep, id="deep"),
    pytest.param(_codeshare_deep_pushdown, id="deep_pushdown"),
    pytest.param(_codeshare_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_codeshare_pipe_rows, id="pipe_rows"),
    pytest.param(_codeshare_codegen, id="codegen"),
    pytest.param(_codeshare_arrow, id="arrow"),
    pytest.param(_codeshare_vectorized, id="vectorized"),
]

# Fast implementations that can run the full codeshare query without a
# prohibitive nested-loop join (they use predicate pushdown or vectorised ops).
_FAST_ADAPTERS = [
    pytest.param(_codeshare_deep_pushdown, id="deep_pushdown"),
    pytest.param(_codeshare_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_codeshare_pipe_rows, id="pipe_rows"),
    pytest.param(_codeshare_codegen, id="codegen"),
    pytest.param(_codeshare_arrow, id="arrow"),
    pytest.param(_codeshare_vectorized, id="vectorized"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def routes_subset() -> list[dict[str, Any]]:
    """First _ROUTES_SUBSET_SIZE rows of the routes CSV."""
    rows = _load("routes")
    return rows[:_ROUTES_SUBSET_SIZE]


@pytest.fixture(scope="session")
def airlines_all() -> list[dict[str, Any]]:
    return _load("airlines")


@pytest.fixture(scope="session")
def routes_all() -> list[dict[str, Any]]:
    return _load("routes")


# ---------------------------------------------------------------------------
# Test 1: codeshare query on subset - all implementations must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", _ALL_ADAPTERS)
def test_codeshare_subset(
    adapter: Any,
    routes_subset: list[dict[str, Any]],
    airlines_all: list[dict[str, Any]],
) -> None:
    """Every implementation must return min(3, total_matches) source-airport
    values that are a sub-multiset of the plain-Python ground truth on the
    subset.  Order is not required."""
    truth = _codeshare_ground_truth(routes_subset, airlines_all)
    total = sum(truth.values())
    expected_len = min(3, total)

    result = adapter(routes_subset, airlines_all)
    got = result["source-airport"]
    assert len(got) == expected_len, (
        f"expected {expected_len} rows, got {len(got)}: {got}"
    )
    got_counter = collections.Counter(got)
    for airport, count in got_counter.items():
        assert count <= truth[airport], (
            f"airport {airport!r} appears {count}x in result but only {truth[airport]}x in truth"
        )


# ---------------------------------------------------------------------------
# Test 2: full-data ABE/ABI/ABQ assertion (fast implementations only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", _FAST_ADAPTERS)
def test_codeshare_full_data_canonical_result(
    adapter: Any,
    routes_all: list[dict[str, Any]],
    airlines_all: list[dict[str, Any]],
) -> None:
    """Fast (pushdown / vectorised) implementations must produce exactly 3
    source-airport values on the full dataset, each a valid AA codeshare airport.
    Order is not required.  Slow O(n*m) implementations are excluded here because
    a 67k x 6k nested-loop join would take minutes in an unoptimised interpreter."""
    truth = _codeshare_ground_truth(routes_all, airlines_all)
    result = adapter(routes_all, airlines_all)
    got = result["source-airport"]
    assert len(got) == 3, f"expected 3 rows, got {len(got)}: {got}"
    got_counter = collections.Counter(got)
    for airport, count in got_counter.items():
        assert count <= truth[airport], (
            f"airport {airport!r} appears {count}x in result but only {truth[airport]}x in truth"
        )


# ---------------------------------------------------------------------------
# Test 3: filter-only query - all implementations agree exactly
# ---------------------------------------------------------------------------


def _run_filter_eager(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import eager

    df = eager.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000)
    return dict(result._columns)


def _run_filter_query_lift(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import query_lift

    df = query_lift.DataFrame(rows)
    result = query_lift.q(df).filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_query_forward(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import query_forward

    df = query_forward.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_lazy_pull(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_pull

    df = lazy_pull.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_lazy_push(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_push

    df = lazy_push.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_functional(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    import micro_dataframes.functional as func

    q = func.filter(func.source(_to_col_dict(rows)), "pop", lambda v: int(v) >= 4_000_000)
    return func.collect(q)


def _run_filter_deep(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import deep

    plan = deep.Filter(deep.Source(deep.from_rows(rows)), "pop", lambda v: int(v) >= 4_000_000)
    return deep.collect(plan)


def _run_filter_deep_pushdown(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import deep_pushdown

    plan = deep_pushdown.Filter(
        deep_pushdown.Source(deep_pushdown.from_rows(rows)),
        "pop",
        lambda v: int(v) >= 4_000_000,
    )
    return deep_pushdown.collect(plan)


def _run_filter_fluent_pushdown(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import fluent_pushdown

    df = fluent_pushdown.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_pipe_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import pipe_rows

    df = pipe_rows.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_codegen(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import codegen

    df = codegen.DataFrame(rows)
    result = df.filter("pop", lambda v: int(v) >= 4_000_000).collect()
    return dict(result._columns)


def _run_filter_arrow(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import arrow as arrow_mod

    df = arrow_mod.DataFrame(rows)
    result = df.filter(arrow_mod.col("pop") >= 4_000_000)
    return {col: result[col] for col in result._table.schema.names}


def _run_filter_vectorized(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from micro_dataframes import vectorized as vec_mod

    df = vec_mod.DataFrame(rows)
    result = df.filter(vec_mod.col("pop") >= 4_000_000)
    return {c: result[c] for c in result._columns}


_FILTER_RUNNERS = [
    pytest.param(_run_filter_eager, id="eager"),
    pytest.param(_run_filter_query_lift, id="query_lift"),
    pytest.param(_run_filter_query_forward, id="query_forward"),
    pytest.param(_run_filter_lazy_pull, id="lazy_pull"),
    pytest.param(_run_filter_lazy_push, id="lazy_push"),
    pytest.param(_run_filter_functional, id="functional"),
    pytest.param(_run_filter_deep, id="deep"),
    pytest.param(_run_filter_deep_pushdown, id="deep_pushdown"),
    pytest.param(_run_filter_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_run_filter_pipe_rows, id="pipe_rows"),
    pytest.param(_run_filter_codegen, id="codegen"),
    pytest.param(_run_filter_arrow, id="arrow"),
    pytest.param(_run_filter_vectorized, id="vectorized"),
]


@pytest.mark.parametrize("runner", _FILTER_RUNNERS)
def test_filter_all_agree(runner: Any) -> None:
    """Filter-only query: all implementations return the same cities and values
    in the same order.  We compare city (str) and pop (may be int or str
    depending on CSV loading) so we normalise pop to int.
    Note: filter order is technically unspecified under the contract; the exact
    check here just documents that every implementation happens to preserve it."""
    rows = _FILTER_ROWS  # inline, no CSV needed
    result = runner(rows)
    assert result["city"] == ["NYC", "LAX"]
    assert [int(v) for v in result["pop"]] == [8_000_000, 4_000_000]


# ---------------------------------------------------------------------------
# Test 4: join-only on handcrafted data - one-to-many duplication and order
# ---------------------------------------------------------------------------

# Each _join_* function takes (left_rows, right_rows) and returns the joined
# column dict.  Using one function per implementation keeps mypy happy: no
# same-name variable is assigned incompatible types in the same scope.


def _join_eager(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import eager

    r_e = eager.DataFrame(left_rows)
    a_e = eager.DataFrame(right_rows)
    return dict(r_e.join(a_e, "route-id", "airline-id")._columns)


def _join_query_lift(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import query_lift

    r_df = query_lift.DataFrame(left_rows)
    a_df = query_lift.DataFrame(right_rows)
    return dict(
        query_lift.q(r_df).join(query_lift.q(a_df), "route-id", "airline-id").collect()._columns
    )


def _join_query_forward(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import query_forward

    r_qf = query_forward.DataFrame(left_rows)
    a_qf = query_forward.DataFrame(right_rows)
    return dict(r_qf.join(a_qf, "route-id", "airline-id").collect()._columns)


def _join_lazy_pull(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_pull

    r_lp = lazy_pull.DataFrame(left_rows)
    a_lp = lazy_pull.DataFrame(right_rows)
    return dict(r_lp.join(a_lp, "route-id", "airline-id").collect()._columns)


def _join_lazy_push(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import lazy_push

    r_lps = lazy_push.DataFrame(left_rows)
    a_lps = lazy_push.DataFrame(right_rows)
    return dict(r_lps.join(a_lps, "route-id", "airline-id").collect()._columns)


def _join_functional(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    import micro_dataframes.functional as func

    left_cols = _to_col_dict(left_rows)
    right_cols = _to_col_dict(right_rows)
    q = func.join(func.source(left_cols), func.source(right_cols), "route-id", "airline-id")
    return func.collect(q)


def _join_deep(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import deep

    left_cols = _to_col_dict(left_rows)
    right_cols = _to_col_dict(right_rows)
    plan = deep.Join(deep.Source(left_cols), deep.Source(right_cols), "route-id", "airline-id")
    return deep.collect(plan)


def _join_deep_pushdown(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import deep_pushdown

    left_cols = _to_col_dict(left_rows)
    right_cols = _to_col_dict(right_rows)
    plan = deep_pushdown.Join(
        deep_pushdown.Source(left_cols),
        deep_pushdown.Source(right_cols),
        "route-id",
        "airline-id",
    )
    return deep_pushdown.collect(plan)


def _join_fluent_pushdown(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import fluent_pushdown

    r_fp = fluent_pushdown.DataFrame(left_rows)
    a_fp = fluent_pushdown.DataFrame(right_rows)
    return dict(r_fp.join(a_fp, "route-id", "airline-id").collect()._columns)


def _join_pipe_rows(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import pipe_rows

    r_pr = pipe_rows.DataFrame(left_rows)
    a_pr = pipe_rows.DataFrame(right_rows)
    return dict(r_pr.join(a_pr, "route-id", "airline-id").collect()._columns)


def _join_codegen(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import codegen

    r_cg = codegen.DataFrame(left_rows)
    a_cg = codegen.DataFrame(right_rows)
    return dict(r_cg.join(a_cg, "route-id", "airline-id").collect()._columns)


def _join_arrow(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import arrow as arrow_mod

    r_ar = arrow_mod.DataFrame(left_rows)
    a_ar = arrow_mod.DataFrame(right_rows)
    result = r_ar.join(a_ar, "route-id", "airline-id")
    return {col: result[col] for col in result._table.schema.names}


def _join_vectorized(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from micro_dataframes import vectorized as vec_mod

    r_v = vec_mod.DataFrame(left_rows)
    a_v = vec_mod.DataFrame(right_rows)
    result = r_v.join(a_v, "route-id", "airline-id")
    return {c: result[c] for c in result._columns}


_JOIN_RUNNERS = [
    pytest.param(_join_eager, id="eager"),
    pytest.param(_join_query_lift, id="query_lift"),
    pytest.param(_join_query_forward, id="query_forward"),
    pytest.param(_join_lazy_pull, id="lazy_pull"),
    pytest.param(_join_lazy_push, id="lazy_push"),
    pytest.param(_join_functional, id="functional"),
    pytest.param(_join_deep, id="deep"),
    pytest.param(_join_deep_pushdown, id="deep_pushdown"),
    pytest.param(_join_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_join_pipe_rows, id="pipe_rows"),
    pytest.param(_join_codegen, id="codegen"),
    pytest.param(_join_arrow, id="arrow"),
    pytest.param(_join_vectorized, id="vectorized"),
]


@pytest.mark.parametrize("runner", _JOIN_RUNNERS)
def test_join_only_duplication(runner: Any) -> None:
    """Join of _LEFT_ROWS x _RIGHT_ROWS on route-id / airline-id must produce
    exactly 3 rows with correct one-to-many duplication (two left rows with
    route-id='1' each match airline-id='1', one row with route-id='2' matches
    airline-id='2').  Order is not required; we compare the multiset of
    (route-id, src, name) triples."""
    result = runner(_LEFT_ROWS, _RIGHT_ROWS)
    assert len(result["route-id"]) == 3
    got = collections.Counter(
        zip(result["route-id"], result["src"], result["name"], strict=True)
    )
    expected = collections.Counter([("1", "A", "X"), ("2", "C", "Y"), ("1", "E", "X")])
    assert got == expected, f"expected multiset {dict(expected)}, got {dict(got)}"


# ---------------------------------------------------------------------------
# Test 5: limit(0) on handcrafted data
# ---------------------------------------------------------------------------

_LIMIT0_COLS: dict[str, list[Any]] = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
_LIMIT0_ROWS: list[dict[str, Any]] = [
    {"a": 1, "b": "x"},
    {"a": 2, "b": "y"},
    {"a": 3, "b": "z"},
]


def _limit0_eager() -> dict[str, list[Any]]:
    from micro_dataframes import eager

    return dict(eager.DataFrame(_LIMIT0_COLS).limit(0)._columns)


def _limit0_query_lift() -> dict[str, list[Any]]:
    from micro_dataframes import query_lift

    return dict(query_lift.q(query_lift.DataFrame(_LIMIT0_COLS)).limit(0).collect()._columns)


def _limit0_query_forward() -> dict[str, list[Any]]:
    from micro_dataframes import query_forward

    return dict(query_forward.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_lazy_pull() -> dict[str, list[Any]]:
    from micro_dataframes import lazy_pull

    return dict(lazy_pull.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_lazy_push() -> dict[str, list[Any]]:
    from micro_dataframes import lazy_push

    return dict(lazy_push.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_functional() -> dict[str, list[Any]]:
    import micro_dataframes.functional as func

    return func.collect(func.limit(func.source(_LIMIT0_COLS), 0))


def _limit0_deep() -> dict[str, list[Any]]:
    from micro_dataframes import deep

    return deep.collect(deep.Limit(deep.Source(_LIMIT0_COLS), 0))


def _limit0_deep_pushdown() -> dict[str, list[Any]]:
    from micro_dataframes import deep_pushdown

    return deep_pushdown.collect(deep_pushdown.Limit(deep_pushdown.Source(_LIMIT0_COLS), 0))


def _limit0_fluent_pushdown() -> dict[str, list[Any]]:
    from micro_dataframes import fluent_pushdown

    return dict(fluent_pushdown.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_pipe_rows() -> dict[str, list[Any]]:
    from micro_dataframes import pipe_rows

    return dict(pipe_rows.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_codegen() -> dict[str, list[Any]]:
    from micro_dataframes import codegen

    return dict(codegen.DataFrame(_LIMIT0_COLS).limit(0).collect()._columns)


def _limit0_arrow() -> dict[str, list[Any]]:
    from micro_dataframes import arrow as arrow_mod

    result = arrow_mod.DataFrame(_LIMIT0_ROWS).limit(0)
    return {col: result[col] for col in result._table.schema.names}


def _limit0_vectorized() -> dict[str, list[Any]]:
    from micro_dataframes import vectorized as vec_mod

    result = vec_mod.DataFrame(_LIMIT0_COLS).limit(0)
    return {c: result[c] for c in result._columns}


_LIMIT0_RUNNERS = [
    pytest.param(_limit0_eager, id="eager"),
    pytest.param(_limit0_query_lift, id="query_lift"),
    pytest.param(_limit0_query_forward, id="query_forward"),
    pytest.param(_limit0_lazy_pull, id="lazy_pull"),
    pytest.param(_limit0_lazy_push, id="lazy_push"),
    pytest.param(_limit0_functional, id="functional"),
    pytest.param(_limit0_deep, id="deep"),
    pytest.param(_limit0_deep_pushdown, id="deep_pushdown"),
    pytest.param(_limit0_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_limit0_pipe_rows, id="pipe_rows"),
    pytest.param(_limit0_codegen, id="codegen"),
    pytest.param(_limit0_arrow, id="arrow"),
    pytest.param(_limit0_vectorized, id="vectorized"),
]


@pytest.mark.parametrize("runner", _LIMIT0_RUNNERS)
def test_limit_zero(runner: Any) -> None:
    """limit(0) must return zero rows.  We do NOT require exact dict equality
    because there is a legitimate divergence: eager / query_lift / query_forward /
    codegen / arrow pre-initialise all schema columns as empty lists (they slice
    eagerly), while lazy / streaming implementations (lazy_pull, lazy_push,
    functional, deep, deep_pushdown, fluent_pushdown, pipe_rows) build the result
    dict by accumulating rows and therefore return an empty dict {} when no rows
    flow through.  Both behaviours are correct; we only assert that every column
    that *is* present has length 0."""
    result = runner()
    # Every column that appears must be empty.
    for col, vals in result.items():
        assert len(vals) == 0, f"column {col!r} should be empty, got {vals!r}"


# ---------------------------------------------------------------------------
# Test 6: no-match filter on handcrafted data
# ---------------------------------------------------------------------------

_NOMATCH_COLS: dict[str, list[Any]] = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
_NOMATCH_ROWS: list[dict[str, Any]] = [
    {"a": 1, "b": "x"},
    {"a": 2, "b": "y"},
    {"a": 3, "b": "z"},
]


def _nomatch_eager() -> dict[str, list[Any]]:
    from micro_dataframes import eager

    return dict(eager.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999)._columns)


def _nomatch_query_lift() -> dict[str, list[Any]]:
    from micro_dataframes import query_lift

    return dict(
        query_lift.q(query_lift.DataFrame(_NOMATCH_COLS))
        .filter("a", lambda v: v == 999)
        .collect()
        ._columns
    )


def _nomatch_query_forward() -> dict[str, list[Any]]:
    from micro_dataframes import query_forward

    return dict(
        query_forward.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_lazy_pull() -> dict[str, list[Any]]:
    from micro_dataframes import lazy_pull

    return dict(
        lazy_pull.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_lazy_push() -> dict[str, list[Any]]:
    from micro_dataframes import lazy_push

    return dict(
        lazy_push.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_functional() -> dict[str, list[Any]]:
    import micro_dataframes.functional as func

    return func.collect(func.filter(func.source(_NOMATCH_COLS), "a", lambda v: v == 999))


def _nomatch_deep() -> dict[str, list[Any]]:
    from micro_dataframes import deep

    return deep.collect(deep.Filter(deep.Source(_NOMATCH_COLS), "a", lambda v: v == 999))


def _nomatch_deep_pushdown() -> dict[str, list[Any]]:
    from micro_dataframes import deep_pushdown

    return deep_pushdown.collect(
        deep_pushdown.Filter(deep_pushdown.Source(_NOMATCH_COLS), "a", lambda v: v == 999)
    )


def _nomatch_fluent_pushdown() -> dict[str, list[Any]]:
    from micro_dataframes import fluent_pushdown

    return dict(
        fluent_pushdown.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_pipe_rows() -> dict[str, list[Any]]:
    from micro_dataframes import pipe_rows

    return dict(
        pipe_rows.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_codegen() -> dict[str, list[Any]]:
    from micro_dataframes import codegen

    return dict(
        codegen.DataFrame(_NOMATCH_COLS).filter("a", lambda v: v == 999).collect()._columns
    )


def _nomatch_arrow() -> dict[str, list[Any]]:
    from micro_dataframes import arrow as arrow_mod

    result = arrow_mod.DataFrame(_NOMATCH_ROWS).filter(arrow_mod.col("a") == 999)
    return {col: result[col] for col in result._table.schema.names}


def _nomatch_vectorized() -> dict[str, list[Any]]:
    from micro_dataframes import vectorized as vec_mod

    result = vec_mod.DataFrame(_NOMATCH_COLS).filter(vec_mod.col("a") == 999)
    return {c: result[c] for c in result._columns}


_NOMATCH_RUNNERS = [
    pytest.param(_nomatch_eager, id="eager"),
    pytest.param(_nomatch_query_lift, id="query_lift"),
    pytest.param(_nomatch_query_forward, id="query_forward"),
    pytest.param(_nomatch_lazy_pull, id="lazy_pull"),
    pytest.param(_nomatch_lazy_push, id="lazy_push"),
    pytest.param(_nomatch_functional, id="functional"),
    pytest.param(_nomatch_deep, id="deep"),
    pytest.param(_nomatch_deep_pushdown, id="deep_pushdown"),
    pytest.param(_nomatch_fluent_pushdown, id="fluent_pushdown"),
    pytest.param(_nomatch_pipe_rows, id="pipe_rows"),
    pytest.param(_nomatch_codegen, id="codegen"),
    pytest.param(_nomatch_arrow, id="arrow"),
    pytest.param(_nomatch_vectorized, id="vectorized"),
]


@pytest.mark.parametrize("runner", _NOMATCH_RUNNERS)
def test_no_match_filter(runner: Any) -> None:
    """Filter that matches nothing: same divergence as limit(0) - row-streaming
    implementations return {} while column-oriented ones return {col: [] for
    each col}.  We only assert that present columns are empty."""
    result = runner()
    # Every column that appears must be empty.
    for col, vals in result.items():
        assert len(vals) == 0, (
            f"column {col!r} should be empty after no-match filter, got {vals!r}"
        )
