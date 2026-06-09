from collections.abc import Callable, Iterator
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


Iter = Callable[[], Iterator[dict[str, Any]]]

class IntermediateResult(Query):
    _iter: Iter

    def __init__(self, make_iter: Iter) -> None:
        self._iter = make_iter

    @classmethod
    def lift(cls, query: Query) -> IntermediateResult:
        if isinstance(query, IntermediateResult):
            return query
        columns = query.collect()._columns
        def make_iter() -> Iterator[dict[str, Any]]:
            n = len(next(iter(columns.values())))
            for i in range(n):
                yield {col: vals[i] for col, vals in columns.items()}
        return cls(make_iter)

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        def make_iter() -> Iterator[dict[str, Any]]:
            for row in self._iter():
                if predicate(row[column]):
                    yield row
        return IntermediateResult(make_iter)

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        def make_iter() -> Iterator[dict[str, Any]]:
            right_rows = list(IntermediateResult.lift(other)._iter())
            for left_row in self._iter():
                for right_row in right_rows:
                    if left_row[left_on] == right_row[right_on]:
                        yield left_row | right_row
        return IntermediateResult(make_iter)

    def limit(self, n: int) -> IntermediateResult:
        def make_iter() -> Iterator[dict[str, Any]]:
            for i, row in enumerate(self._iter()):
                if i == n:
                    break
                yield row
        return IntermediateResult(make_iter)

    def collect(self) -> DataFrame:
        columns: dict[str, list[Any]] = {}
        for row in self._iter():
            for col, val in row.items():
                columns.setdefault(col, []).append(val)
        return DataFrame(columns)
