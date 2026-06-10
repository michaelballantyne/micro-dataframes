# Regression tests for join corner cases under the relaxed contract:
#   * Row order is undefined — limit(n) may return ANY n rows.
#   * Column names of joined tables are assumed disjoint.
#
# What is still tested:
#   * A limit on the right side of a join: the result must have the right
#     cardinality, each row must be internally consistent, and the two rows
#     must be distinct.  This catches the original codegen bug where a limit
#     inside the build pipeline returned the empty result.

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

# The three valid full-join rows (lid, val, name) for reference.
_VALID_ROWS = {("1", "a", "X"), ("2", "b", "Y"), ("3", "c", "Z")}


@pytest.mark.parametrize("impl", _ALL)
def test_limit_on_right_side_of_join(impl: str) -> None:
    # The right side is limited to any 2 of its 3 rows before the join.
    # Under the relaxed contract (order undefined), we only assert:
    #   1. Exactly 2 output rows.
    #   2. Each output row is internally consistent: lid == rid at that
    #      position, and the (lid, val, name) triple is one of the valid rows.
    #   3. The two output rows are distinct.
    # This catches the codegen bug where a limit inside the build pipeline
    # caused the buffer to be empty, returning zero rows.
    result = _join(impl, _LEFT, _RIGHT, "lid", "rid", limit_right=2)

    lids = result["lid"]
    rids = result["rid"]
    vals = result["val"]
    names = result["name"]

    assert len(lids) == 2, f"expected 2 rows, got {len(lids)}"

    for i in range(2):
        assert lids[i] == rids[i], (
            f"row {i}: lid={lids[i]!r} != rid={rids[i]!r} — row is inconsistent"
        )
        triple = (lids[i], vals[i], names[i])
        assert triple in _VALID_ROWS, f"row {i}: {triple!r} is not a valid join row"

    row0 = (lids[0], vals[0], names[0])
    row1 = (lids[1], vals[1], names[1])
    assert row0 != row1, "the two output rows must be distinct"
