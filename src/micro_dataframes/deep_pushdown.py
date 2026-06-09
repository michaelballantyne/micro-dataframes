from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

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


def from_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


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
            right_rows = list(execute(right))
            for left_row in execute(left):
                for right_row in right_rows:
                    if left_row[left_on] == right_row[right_on]:
                        yield left_row | right_row
        case Limit(child, n):
            for i, row in enumerate(execute(child)):
                if i == n:
                    break
                yield row


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


def collect(plan: Plan) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for row in execute(optimize(plan)):
        for col, val in row.items():
            columns.setdefault(col, []).append(val)
    return columns
