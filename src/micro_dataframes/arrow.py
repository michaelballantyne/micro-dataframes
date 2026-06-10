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
        # The other implementations merge rows with `left_row | right_row`:
        # on a column name collision the right side wins, and both key
        # columns appear when their names differ.  pyarrow's hash-join would
        # instead produce duplicate column names, and it does not preserve
        # nested-loop order (left rows in original order, each followed by
        # its matches in right order).  Sentinel columns repair both: integer
        # row indices to sort by afterward, and a renamed left key so left
        # columns the right side also has can be dropped before the join.
        sentinel_l = "__left_row_idx__"
        sentinel_r = "__right_row_idx__"
        left = self._table.append_column(
            sentinel_l, pa.array(range(self._table.num_rows), type=pa.int64())
        )
        right = other._table.append_column(
            sentinel_r, pa.array(range(other._table.num_rows), type=pa.int64())
        )

        key = left_on
        if left_on in other._table.column_names:
            key = "__left_key__"
            left = left.rename_columns(
                [key if name == left_on else name for name in left.column_names]
            )
        left = left.drop_columns(
            [name for name in left.column_names if name in set(other._table.column_names)]
        )

        joined = left.join(
            right,
            keys=key,
            right_keys=right_on,
            join_type="inner",
            coalesce_keys=False,  # keep both key columns so right_on survives
        )
        sorted_joined = joined.sort_by(
            [(sentinel_l, "ascending"), (sentinel_r, "ascending")]
        )
        dropped = [sentinel_l, sentinel_r] + ([key] if key != left_on else [])
        return DataFrame._from_table(sorted_joined.drop_columns(dropped))

    def limit(self, n: int) -> DataFrame:
        return DataFrame._from_table(self._table.slice(0, n))

    def __getitem__(self, column: str) -> list[Any]:
        return self._table.column(column).to_pylist()
