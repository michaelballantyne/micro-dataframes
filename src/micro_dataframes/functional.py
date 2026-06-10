from collections.abc import Callable, Iterator
from typing import Any

# A query is just a thunk producing rows. The representation is the
# public interface: user-defined operators are written exactly like the
# ones below, with no privileged access.
type Row = dict[str, Any]
type Query = Callable[[], Iterator[Row]]


def from_rows(rows: list[Row]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def source(columns: dict[str, list[Any]]) -> Query:
    def rows() -> Iterator[Row]:
        n = len(next(iter(columns.values())))
        for i in range(n):
            yield {col: vals[i] for col, vals in columns.items()}
    return rows


def filter(q: Query, column: str, predicate: Callable[[Any], bool]) -> Query:
    def rows() -> Iterator[Row]:
        for row in q():
            if predicate(row[column]):
                yield row
    return rows


def join(left: Query, right: Query, left_on: str, right_on: str) -> Query:
    def rows() -> Iterator[Row]:
        # Build: index the right side by key.  Probe: stream the left side.
        index: dict[Any, list[Row]] = {}
        for right_row in right():
            index.setdefault(right_row[right_on], []).append(right_row)
        for left_row in left():
            for right_row in index.get(left_row[left_on], []):
                yield left_row | right_row
    return rows


def limit(q: Query, n: int) -> Query:
    def rows() -> Iterator[Row]:
        for i, row in enumerate(q()):
            if i == n:
                break
            yield row
    return rows


def collect(q: Query) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for row in q():
        for col, val in row.items():
            columns.setdefault(col, []).append(val)
    return columns


def thread(q: Query, *steps: Callable[[Query], Query]) -> Query:
    for step in steps:
        q = step(q)
    return q
