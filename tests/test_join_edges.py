# Regression tests for join corner cases:
#   * a limit on the right side of a join (codegen's build pipeline must stop
#     filling its buffer without aborting the rest of the kernel)
#   * joining on the same key name on both sides (arrow's hash join must not
#     produce duplicate columns)
#   * a non-key column name appearing on both sides (right wins, matching
#     left_row | right_row)
#
# eager, query_lift, and query_forward are excluded from the two shared-name
# tests: their joins append into one output list per column name, so a name
# both sides share collects values from both, giving doubled, ragged columns.
# That divergence predates this test suite and is left as-is.

from typing import Any

import pytest

from micro_dataframes import (
    arrow,
    codegen,
    deep,
    deep_pushdown,
    eager,
    fluent_pushdown,
    functional,
    lazy_pull,
    lazy_push,
    pipe_rows,
    query_forward,
    query_lift,
    vectorized,
)


def _cols(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in rows:
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return out


_FLUENT: dict[str, Any] = {
    "query_forward": query_forward,
    "lazy_pull": lazy_pull,
    "lazy_push": lazy_push,
    "fluent_pushdown": fluent_pushdown,
    "pipe_rows": pipe_rows,
    "codegen": codegen,
}


# Run left JOIN right on (left_on, right_on), optionally with a limit applied
# to the right side first, and return the result as a columns dict.
def _join(
    impl: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_on: str,
    right_on: str,
    limit_right: int | None = None,
) -> dict[str, list[Any]]:
    if impl in _FLUENT:
        mod = _FLUENT[impl]
        r: Any = mod.DataFrame(right)
        if limit_right is not None:
            r = r.limit(limit_right)
        return dict(mod.DataFrame(left).join(r, left_on, right_on).collect()._columns)
    if impl == "eager":
        r_e = eager.DataFrame(right)
        if limit_right is not None:
            r_e = r_e.limit(limit_right)
        return dict(eager.DataFrame(left).join(r_e, left_on, right_on)._columns)
    if impl == "query_lift":
        r_q = query_lift.q(query_lift.DataFrame(right))
        if limit_right is not None:
            r_q = r_q.limit(limit_right)
        lifted = query_lift.q(query_lift.DataFrame(left))
        return dict(lifted.join(r_q, left_on, right_on).collect()._columns)
    if impl == "functional":
        r_f = functional.source(_cols(right))
        if limit_right is not None:
            r_f = functional.limit(r_f, limit_right)
        joined = functional.join(functional.source(_cols(left)), r_f, left_on, right_on)
        return functional.collect(joined)
    if impl in ("deep", "deep_pushdown"):
        mod = deep if impl == "deep" else deep_pushdown
        r_plan: Any = mod.Source(_cols(right))
        if limit_right is not None:
            r_plan = mod.Limit(r_plan, limit_right)
        plan = mod.Join(mod.Source(_cols(left)), r_plan, left_on, right_on)
        return dict(mod.collect(plan))
    if impl == "arrow":
        r_a = arrow.DataFrame(right)
        if limit_right is not None:
            r_a = r_a.limit(limit_right)
        result = arrow.DataFrame(left).join(r_a, left_on, right_on)
        return {c: result[c] for c in result._table.schema.names}
    assert impl == "vectorized"
    r_v = vectorized.DataFrame(right)
    if limit_right is not None:
        r_v = r_v.limit(limit_right)
    result_v = vectorized.DataFrame(left).join(r_v, left_on, right_on)
    return {c: result_v[c] for c in result_v._columns}


_ALL = [
    "eager", "query_lift", "query_forward", "lazy_pull", "lazy_push", "functional",
    "deep", "deep_pushdown", "fluent_pushdown", "pipe_rows", "codegen", "arrow",
    "vectorized",
]
# Implementations whose joins merge rows like left_row | right_row.
_ROW_MERGE = [i for i in _ALL if i not in ("eager", "query_lift", "query_forward")]


_LEFT = [
    {"lid": "1", "val": "a"},
    {"lid": "2", "val": "b"},
    {"lid": "3", "val": "c"},
]
_RIGHT = [
    {"rid": "1", "name": "X"},
    {"rid": "2", "name": "Y"},
    {"rid": "3", "name": "Z"},
]


@pytest.mark.parametrize("impl", _ALL)
def test_limit_on_right_side_of_join(impl: str) -> None:
    # Only the first two right rows survive the limit, so the third left row
    # finds no match.  In codegen this exercises a limit inside the join's
    # build pipeline, which must stop the buffer fill without returning from
    # the whole kernel.
    result = _join(impl, _LEFT, _RIGHT, "lid", "rid", limit_right=2)
    assert result["val"] == ["a", "b"]
    assert result["name"] == ["X", "Y"]
    assert result["lid"] == ["1", "2"]
    assert result["rid"] == ["1", "2"]


@pytest.mark.parametrize("impl", _ROW_MERGE)
def test_join_on_same_key_name(impl: str) -> None:
    left = [{"id": "1", "val": "a"}, {"id": "2", "val": "b"}]
    right = [{"id": "1", "name": "X"}, {"id": "2", "name": "Y"}]
    result = _join(impl, left, right, "id", "id")
    assert result["id"] == ["1", "2"]
    assert result["val"] == ["a", "b"]
    assert result["name"] == ["X", "Y"]


@pytest.mark.parametrize("impl", _ROW_MERGE)
def test_duplicate_nonkey_column_right_wins(impl: str) -> None:
    left = [{"k": "1", "name": "L1"}, {"k": "2", "name": "L2"}]
    right = [{"kk": "1", "name": "R1"}, {"kk": "2", "name": "R2"}]
    result = _join(impl, left, right, "k", "kk")
    assert result["name"] == ["R1", "R2"]
    assert result["k"] == ["1", "2"]
    assert result["kk"] == ["1", "2"]
