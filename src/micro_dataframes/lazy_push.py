from collections.abc import Callable
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


Consumer = Callable[[dict[str, Any]], None]


class _StopPushing(Exception):
    pass


class IntermediateResult(Query):
    _run: Callable[[Consumer], None]

    def __init__(self, run: Callable[[Consumer], None]) -> None:
        self._run = run

    @classmethod
    def lift(cls, query: Query) -> IntermediateResult:
        if isinstance(query, IntermediateResult):
            return query
        columns = query.collect()._columns
        def run(consumer: Consumer) -> None:
            n = len(next(iter(columns.values())))
            for i in range(n):
                consumer({col: vals[i] for col, vals in columns.items()})
        return cls(run)

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        def run(consumer: Consumer) -> None:
            self._run(lambda row: consumer(row) if predicate(row[column]) else None)
        return IntermediateResult(run)

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        def run(consumer: Consumer) -> None:
            right_rows: list[dict[str, Any]] = []
            IntermediateResult.lift(other)._run(right_rows.append)
            def joined(left_row: dict[str, Any]) -> None:
                for right_row in right_rows:
                    if left_row[left_on] == right_row[right_on]:
                        consumer(left_row | right_row)
            self._run(joined)
        return IntermediateResult(run)

    def limit(self, n: int) -> IntermediateResult:
        # A consumer can't tell its producer to stop pushing, so escape
        # with an exception once enough rows have arrived. Contrast with
        # the pull version, where the consumer simply stops pulling.
        def run(consumer: Consumer) -> None:
            count = 0
            def limited(row: dict[str, Any]) -> None:
                nonlocal count
                if count == n:
                    raise _StopPushing()
                count += 1
                consumer(row)
            try:
                self._run(limited)
            except _StopPushing:
                pass
        return IntermediateResult(run)

    def collect(self) -> DataFrame:
        columns: dict[str, list[Any]] = {}
        def accumulate(row: dict[str, Any]) -> None:
            for col, val in row.items():
                columns.setdefault(col, []).append(val)
        self._run(accumulate)
        return DataFrame(columns)
