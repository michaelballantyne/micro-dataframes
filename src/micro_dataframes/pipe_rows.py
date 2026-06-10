from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

type Rows = Iterator[dict[str, Any]]


class Query(Protocol):
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult: ...
    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult: ...
    def pipe_rows(self, fn: Callable[[Rows], Rows]) -> IntermediateResult: ...
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

    def pipe_rows(self, fn: Callable[[Rows], Rows]) -> IntermediateResult:
        return IntermediateResult.lift(self).pipe_rows(fn)

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

    def pipe_rows(self, fn: Callable[[Rows], Rows]) -> IntermediateResult:
        return IntermediateResult(PipeRows(self._plan, fn))

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult(Limit(self._plan, n))

    def collect(self) -> DataFrame:
        columns: dict[str, list[Any]] = {}
        for row in execute(optimize(self._plan)):
            for col, val in row.items():
                columns.setdefault(col, []).append(val)
        return DataFrame(columns)


# --- Plan nodes (the deep embedding) ---

type Plan = Source | Filter | Join | PipeRows | Limit


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
class PipeRows:
    child: Plan
    fn: Callable[[Rows], Rows]


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
        case PipeRows(child, _):
            # We can't know what fn does to the rows; assume it preserves
            # the column set. Nothing checks this.
            return schema(child)
        case Limit(child, _):
            return schema(child)


def execute(plan: Plan) -> Iterator[dict[str, Any]]:
    match plan:
        case Source(columns):
            n = len(next(iter(columns.values())))
            for i in range(n):
                yield {col: vals[i] for col, vals in columns.items()}
        case Filter(child, column, predicate):
            for row in execute(child):
                if predicate(row[column]):
                    yield row
        case Join(left, right, left_on, right_on):
            # Build: index the right side by key.  Probe: stream the left side.
            index: dict[Any, list[dict[str, Any]]] = {}
            for right_row in execute(right):
                index.setdefault(right_row[right_on], []).append(right_row)
            for left_row in execute(left):
                for right_row in index.get(left_row[left_on], []):
                    yield left_row | right_row
        case PipeRows(child, fn):
            yield from fn(execute(child))
        case Limit(child, n):
            for i, row in enumerate(execute(child)):
                if i == n:
                    break
                yield row


def optimize(plan: Plan) -> Plan:
    match plan:
        case Filter(child, column, predicate):
            # Push a filter below a join when its column comes from one side.
            # A filter never moves below a PipeRows: fn is opaque and may be
            # stateful (e.g. distinct), so reordering would be unsound.
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
        case PipeRows(child, fn):
            return PipeRows(optimize(child), fn)
        case Limit(child, n):
            return Limit(optimize(child), n)
        case Source():
            return plan
