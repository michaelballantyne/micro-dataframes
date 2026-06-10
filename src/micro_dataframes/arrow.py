from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

# --- Expression DSL ---
#
# Arrow's compute backend cannot execute opaque Python lambdas column-wise.
# Instead we expose a small DSL whose operators build pc.Expression trees,
# which pyarrow can evaluate natively.  This is a shallow embedding: each
# Expr node holds a single pc.Expression value and simply delegates to it.


@dataclass(frozen=True)
class Expr:
    """A column expression that translates to a pyarrow.compute.Expression."""

    _expr: pc.Expression

    # Comparisons — return Expr, not bool.
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(self._expr == other)

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(self._expr != other)

    def __lt__(self, other: object) -> Expr:
        return Expr(self._expr < other)

    def __le__(self, other: object) -> Expr:
        return Expr(self._expr <= other)

    def __gt__(self, other: object) -> Expr:
        return Expr(self._expr > other)

    def __ge__(self, other: object) -> Expr:
        return Expr(self._expr >= other)

    # Boolean combinators.
    def __and__(self, other: Expr) -> Expr:
        return Expr(self._expr & other._expr)

    def __or__(self, other: Expr) -> Expr:
        return Expr(self._expr | other._expr)

    def __invert__(self) -> Expr:
        return Expr(~self._expr)

    def __hash__(self) -> int:
        return hash(str(self._expr))


def col(name: str) -> Expr:
    """Reference a column by name, returning an Expr for building filter predicates."""
    return Expr(pc.field(name))


# --- DataFrame ---


class DataFrame:
    _table: pa.Table

    def __init__(self, data: dict[str, list[Any]] | list[dict[str, Any]]) -> None:
        if isinstance(data, list):
            self._table = pa.Table.from_pylist(data)
        else:
            self._table = pa.table(data)

    @classmethod
    def _from_table(cls, table: pa.Table) -> DataFrame:
        obj = cls.__new__(cls)
        obj._table = table
        return obj

    def filter(self, expr: Expr) -> DataFrame:
        return DataFrame._from_table(self._table.filter(expr._expr))

    def join(self, other: DataFrame, left_on: str, right_on: str) -> DataFrame:
        # pyarrow's hash-join does not preserve nested-loop order (left rows in
        # original order, each followed by matching right rows in original order).
        # Fix: attach integer row-index columns to both sides, join, sort by
        # (left_idx, right_idx), then drop the sentinel columns.
        sentinel_l = "__left_row_idx__"
        sentinel_r = "__right_row_idx__"
        left = self._table.append_column(
            sentinel_l, pa.array(range(self._table.num_rows), type=pa.int64())
        )
        right = other._table.append_column(
            sentinel_r, pa.array(range(other._table.num_rows), type=pa.int64())
        )

        joined = left.join(
            right,
            keys=left_on,
            right_keys=right_on,
            join_type="inner",
            coalesce_keys=False,  # keep both key columns so right_on survives
        )
        sorted_joined = joined.sort_by(
            [(sentinel_l, "ascending"), (sentinel_r, "ascending")]
        )
        return DataFrame._from_table(
            sorted_joined.drop_columns([sentinel_l, sentinel_r])
        )

    def limit(self, n: int) -> DataFrame:
        return DataFrame._from_table(self._table.slice(0, n))

    def __getitem__(self, column: str) -> list[Any]:
        return self._table.column(column).to_pylist()
