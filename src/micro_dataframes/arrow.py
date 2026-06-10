from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

# --- Expression DSL ---
#
# Arrow's compute backend cannot execute opaque Python lambdas column-wise.
# Instead we expose a small DSL whose operators build pc.Expression trees,
# which pyarrow can evaluate natively.  This is a shallow embedding: each
# Expr node holds a single pc.Expression value and simply delegates to it.


def _unwrap(other: object) -> object:
    # If the right-hand operand is itself an Expr, peel off the pc.Expression
    # so that pyarrow receives two pc.Expression objects (column-to-column).
    return other._expr if isinstance(other, Expr) else other


class Expr:
    def __init__(self, expr: pc.Expression) -> None:
        self._expr = expr

    # Comparisons — return Expr, not bool.
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(self._expr == _unwrap(other))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(self._expr != _unwrap(other))

    def __lt__(self, other: object) -> Expr:
        return Expr(self._expr < _unwrap(other))

    def __le__(self, other: object) -> Expr:
        return Expr(self._expr <= _unwrap(other))

    def __gt__(self, other: object) -> Expr:
        return Expr(self._expr > _unwrap(other))

    def __ge__(self, other: object) -> Expr:
        return Expr(self._expr >= _unwrap(other))

    # Boolean combinators.
    def __and__(self, other: Expr) -> Expr:
        return Expr(self._expr & other._expr)

    def __or__(self, other: Expr) -> Expr:
        return Expr(self._expr | other._expr)

    def __invert__(self) -> Expr:
        return Expr(~self._expr)


def col(name: str) -> Expr:
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
        # Under the relaxed contract (row order undefined, column names disjoint)
        # a single delegated call is all that is needed — no sentinel columns to
        # restore order, no collision bookkeeping to resolve.
        # coalesce_keys=False: with distinct key names pyarrow would otherwise drop
        # the right key column; the family keeps both.
        return DataFrame._from_table(
            self._table.join(
                other._table,
                keys=left_on,
                right_keys=right_on,
                join_type="inner",
                coalesce_keys=False,
            )
        )

    def limit(self, n: int) -> DataFrame:
        return DataFrame._from_table(self._table.slice(0, n))

    def __getitem__(self, column: str) -> list[Any]:
        return self._table.column(column).to_pylist()
