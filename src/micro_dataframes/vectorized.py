import operator as op
from dataclasses import dataclass
from typing import Any

# --- Predicate DSL (deep embedding) ---
#
# The DSL is a small deep embedding: frozen dataclasses hold the AST nodes,
# `type Pred = ...` declares the union, and `Expr` is a thin wrapper that
# builds nodes via operator overloading.  This mirrors the house style of
# deep.py / deep_pushdown.py but applied to predicates rather than query plans.
#
# Contrast with arrow.py (shallow embedding): there each Expr wraps a live
# pc.Expression that already knows how to evaluate itself.  Here the Expr builds
# a pure data structure; evaluation happens separately in eval_pred.


@dataclass(frozen=True)
class Column:
    name: str


@dataclass(frozen=True)
class Compare:
    # op is one of operator.eq/ne/lt/le/gt/ge — stored as a callable so
    # eval_pred never has to switch on a string.
    left: Pred
    op: Any  # operator function, not inspectable by dataclass eq — that's fine
    right: Pred | object  # right may be a plain Python scalar


@dataclass(frozen=True)
class And:
    left: Pred
    right: Pred


@dataclass(frozen=True)
class Or:
    left: Pred
    right: Pred


@dataclass(frozen=True)
class Not:
    child: Pred


type Pred = Column | Compare | And | Or | Not


def _unwrap(other: object) -> Pred | object:
    # Peel the AST node out of an Expr so that the right-hand side of a
    # comparison can be either another Column node or a plain Python value.
    return other._pred if isinstance(other, Expr) else other


class Expr:
    def __init__(self, pred: Pred) -> None:
        self._pred = pred

    # Comparisons — return Expr, not bool.
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(Compare(self._pred, op.eq, _unwrap(other)))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return Expr(Compare(self._pred, op.ne, _unwrap(other)))

    def __lt__(self, other: object) -> Expr:
        return Expr(Compare(self._pred, op.lt, _unwrap(other)))

    def __le__(self, other: object) -> Expr:
        return Expr(Compare(self._pred, op.le, _unwrap(other)))

    def __gt__(self, other: object) -> Expr:
        return Expr(Compare(self._pred, op.gt, _unwrap(other)))

    def __ge__(self, other: object) -> Expr:
        return Expr(Compare(self._pred, op.ge, _unwrap(other)))

    # Boolean combinators.
    def __and__(self, other: Expr) -> Expr:
        return Expr(And(self._pred, other._pred))

    def __or__(self, other: Expr) -> Expr:
        return Expr(Or(self._pred, other._pred))

    def __invert__(self) -> Expr:
        return Expr(Not(self._pred))


def col(name: str) -> Expr:
    return Expr(Column(name))


# --- Kernels ---
#
# These functions are the ONLY place row-level loops live.  Operators (filter,
# join, limit) compose them; they never iterate rows themselves.


def compare(left: list[Any], cmp_op: Any, right: list[Any] | Any) -> list[bool]:
    # Full-column comparison.  When right is a list the comparison is
    # element-wise (column vs column); otherwise it is column vs scalar.
    if isinstance(right, list):
        return [cmp_op(lv, rv) for lv, rv in zip(left, right, strict=True)]
    return [cmp_op(v, right) for v in left]


def mask_and(a: list[bool], b: list[bool]) -> list[bool]:
    return [av and bv for av, bv in zip(a, b, strict=True)]


def mask_or(a: list[bool], b: list[bool]) -> list[bool]:
    return [av or bv for av, bv in zip(a, b, strict=True)]


def mask_not(a: list[bool]) -> list[bool]:
    return [not v for v in a]


def take(column: list[Any], sel: list[int]) -> list[Any]:
    # Gather: collect the elements of column at positions given by sel.
    return [column[i] for i in sel]


def hash_join(
    left_keys: list[Any],
    left_sel: list[int],
    right_keys: list[Any],
    right_sel: list[int],
) -> tuple[list[int], list[int]]:
    # Build a dict mapping each right key value to the list of right row indices
    # that carry it, iterating right_sel in order so that right-side order is
    # preserved within each bucket.
    index: dict[Any, list[int]] = {}
    for ri in right_sel:
        index.setdefault(right_keys[ri], []).append(ri)

    # Probe: for each left row (in left_sel order), emit one output row per
    # matching right row.  This preserves the nested-loop order that the other
    # implementations use: left rows in left-table order, each left row's
    # matches in right-table order.
    out_left: list[int] = []
    out_right: list[int] = []
    for li in left_sel:
        for ri in index.get(left_keys[li], []):
            out_left.append(li)
            out_right.append(ri)
    return out_left, out_right


def eval_pred(pred: Pred, columns: dict[str, list[Any]]) -> list[Any]:
    # Evaluate a Pred tree over full columns.  Boolean nodes return a mask of
    # length equal to the number of rows; a bare Column returns the column
    # data itself (for column-to-column comparisons).  Kernels run over the
    # whole column; the selection vector in DataFrame.filter decides which
    # rows are live.
    match pred:
        case Column(name):
            return columns[name]
        case Compare(left, cmp_op, right):
            left_vals = eval_pred(left, columns)
            # right may be a nested Pred or a plain Python scalar.
            right_vals: list[Any] | Any = (
                eval_pred(right, columns)
                if isinstance(right, Column | Compare | And | Or | Not)
                else right
            )
            return compare(left_vals, cmp_op, right_vals)
        case And(left, right):
            return mask_and(eval_pred(left, columns), eval_pred(right, columns))
        case Or(left, right):
            return mask_or(eval_pred(left, columns), eval_pred(right, columns))
        case Not(child):
            return mask_not(eval_pred(child, columns))


# --- DataFrame ---


class DataFrame:
    _columns: dict[str, list[Any]]
    _sel: list[int]  # selection vector — indices of currently live rows

    def __init__(self, data: dict[str, list[Any]] | list[dict[str, Any]]) -> None:
        if isinstance(data, list):
            cols: dict[str, list[Any]] = {}
            for row in data:
                for key, value in row.items():
                    if key not in cols:
                        cols[key] = []
                    cols[key].append(value)
            self._columns = cols
        else:
            self._columns = data
        n = len(next(iter(self._columns.values()))) if self._columns else 0
        self._sel = list(range(n))

    @classmethod
    def _from_parts(cls, columns: dict[str, list[Any]], sel: list[int]) -> DataFrame:
        obj = cls.__new__(cls)
        obj._columns = columns
        obj._sel = sel
        return obj

    def filter(self, expr: Expr) -> DataFrame:
        # Evaluate the predicate over the full (unfiltered) columns, then
        # restrict the selection vector to rows where the mask is True.
        # No column data is copied — only the index list shrinks.
        mask = eval_pred(expr._pred, self._columns)
        new_sel = [i for i in self._sel if mask[i]]
        return DataFrame._from_parts(self._columns, new_sel)

    def join(self, other: DataFrame, left_on: str, right_on: str) -> DataFrame:
        # Produce the cross-product index vectors via hash_join, then
        # materialize merged columns with take.  Left columns are gathered
        # first; right columns follow, overwriting on name collision so that
        # right wins — mirroring left_row | right_row in the row-streaming
        # implementations.  When left_on == right_on, both sides emit the same
        # column name and the right value survives (same semantics).
        li_vec, ri_vec = hash_join(
            self._columns[left_on], self._sel,
            other._columns[right_on], other._sel,
        )
        merged: dict[str, list[Any]] = {}
        for name, col_data in self._columns.items():
            merged[name] = take(col_data, li_vec)
        for name, col_data in other._columns.items():
            merged[name] = take(col_data, ri_vec)
        # Result rows are already in the correct order; use a full fresh sel.
        return DataFrame._from_parts(merged, list(range(len(li_vec))))

    def limit(self, n: int) -> DataFrame:
        # Slice the selection vector — no column data copied.
        return DataFrame._from_parts(self._columns, self._sel[:n])

    def __getitem__(self, column: str) -> list[Any]:
        # Apply the selection vector when returning a column.
        return take(self._columns[column], self._sel)
