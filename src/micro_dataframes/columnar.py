from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class Query(Protocol):
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult: ...
    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult: ...
    def limit(self, n: int) -> IntermediateResult: ...
    def collect(self) -> DataFrame: ...


class DataFrame(Query):
    _columns: dict[str, list[Any]]

    def __init__(self, data: dict[str, list[Any]] | list[dict[str, Any]]) -> None:
        if isinstance(data, list):
            self._columns = {}
            for row in data:
                for key, value in row.items():
                    if key not in self._columns:
                        self._columns[key] = []
                    self._columns[key].append(value)
        else:
            self._columns = data

    # Forward Query methods to IntermediateResult.
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        return IntermediateResult.lift(self).filter(column, predicate)

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        return IntermediateResult.lift(self).join(other, left_on, right_on)

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult.lift(self).limit(n)

    # Except collect.
    def collect(self) -> DataFrame:
        return self

    def __getitem__(self, column: str) -> list[Any]:
        return self._columns[column]


class IntermediateResult(Query):
    _plan: Plan

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    @classmethod
    def lift(cls, query: Query) -> IntermediateResult:
        # Keeping an IntermediateResult's plan intact (rather than
        # collecting it) is what lets the optimizer push filters into
        # the right side of a join.
        if isinstance(query, IntermediateResult):
            return query
        return cls(Source(query.collect()._columns))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        return IntermediateResult(Filter(self._plan, column, predicate))

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        other_plan = IntermediateResult.lift(other)._plan
        return IntermediateResult(Join(self._plan, other_plan, left_on, right_on))

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult(Limit(self._plan, n))

    def collect(self) -> DataFrame:
        return DataFrame(execute(optimize(self._plan)))


# --- Plan nodes (the deep embedding) ---

type Plan = Source | Filter | Join | Limit


@dataclass(frozen=True)
class Source:
    columns: dict[str, list[Any]]

    def __repr__(self) -> str:
        return f"Source(columns={sorted(self.columns.keys())})"


@dataclass(frozen=True)
class Filter:
    child: Plan
    column: str
    predicate: Callable[[Any], bool]


@dataclass(frozen=True)
class Join:
    left: Plan
    right: Plan
    left_on: str
    right_on: str


@dataclass(frozen=True)
class Limit:
    child: Plan
    n: int


# --- Passes over plans ---

def schema(plan: Plan) -> set[str]:
    match plan:
        case Source(columns):
            return set(columns.keys())
        case Filter(child, _, _):
            return schema(child)
        case Join(left, right, _, _):
            return schema(left) | schema(right)
        case Limit(child, _):
            return schema(child)


def optimize(plan: Plan) -> Plan:
    match plan:
        case Filter(child, column, predicate):
            # Push a filter below a join when its column comes from one side.
            match optimize(child):
                case Join(left, right, left_on, right_on) if column in schema(left):
                    return Join(optimize(Filter(left, column, predicate)),
                                right, left_on, right_on)
                case Join(left, right, left_on, right_on) if column in schema(right):
                    return Join(left,
                                optimize(Filter(right, column, predicate)),
                                left_on, right_on)
                case child:
                    return Filter(child, column, predicate)
        case Join(left, right, left_on, right_on):
            return Join(optimize(left), optimize(right), left_on, right_on)
        case Limit(child, n):
            return Limit(optimize(child), n)
        case Source():
            return plan


# --- Columnar execution ---
#
# Where the other backends stream rows (dicts) through the plan one at a
# time, this one executes an operator at a time over whole columns: each
# node consumes and produces a dict of column lists. Filter and join work
# with row indices ("selection vectors") and gather the surviving columns
# in one pass per column; join builds a hash index on the build side's key
# column instead of nested-looping over rows. The trade-off shows in
# limit: by the time it runs, its child has already been evaluated in
# full, so limit can only slice -- there is no early exit.
#
# One visible difference from the row engines: an empty result still
# knows its columns. Row-at-a-time collect() only learns the schema from
# rows it sees, so when nothing matches it returns a frame with no
# columns at all; here the column dict flows through whether or not any
# values survive.

def execute(plan: Plan) -> dict[str, list[Any]]:
    match plan:
        case Source(columns):
            return dict(columns)
        case Filter(child, column, predicate):
            columns = execute(child)
            keep = [i for i, value in enumerate(columns[column]) if predicate(value)]
            return {col: [vals[i] for i in keep] for col, vals in columns.items()}
        case Join(left, right, left_on, right_on):
            left_columns = execute(left)
            right_columns = execute(right)
            # Build a hash index on the right key column, then probe it
            # with the left key column to get matching index pairs.
            index: dict[Any, list[int]] = {}
            for j, key in enumerate(right_columns[right_on]):
                index.setdefault(key, []).append(j)
            left_take: list[int] = []
            right_take: list[int] = []
            for i, key in enumerate(left_columns[left_on]):
                for j in index.get(key, ()):
                    left_take.append(i)
                    right_take.append(j)
            joined = {col: [vals[i] for i in left_take]
                      for col, vals in left_columns.items()}
            # On a name collision the right side wins, matching the row
            # engines' `left_row | right_row`.
            joined |= {col: [vals[j] for j in right_take]
                       for col, vals in right_columns.items()}
            return joined
        case Limit(child, n):
            columns = execute(child)
            return {col: vals[:max(n, 0)] for col, vals in columns.items()}
