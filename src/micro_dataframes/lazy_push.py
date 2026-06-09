from typing import Any, Callable, Protocol


Consumer = Callable[[dict[str, Any]], None]


class Query(Protocol):
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "IntermediateResult": ...
    def join(self, other: "Query", left_on: str, right_on: str) -> "IntermediateResult": ...
    def collect(self) -> "DataFrame": ...


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

    def __getitem__(self, column: str) -> list[Any]:
        return self._columns[column]

    # Forward Query methods to IntermediateResult.
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "IntermediateResult":
        return IntermediateResult.from_columns(self._columns).filter(column, predicate)

    def join(self, other: Query, left_on: str, right_on: str) -> "IntermediateResult":
        return IntermediateResult.from_columns(self._columns).join(other, left_on, right_on)

    # Except collect.
    def collect(self) -> "DataFrame":
        return self


class IntermediateResult(Query):
    _run: Callable[[Consumer], None]

    def __init__(self, run: Callable[[Consumer], None]) -> None:
        self._run = run

    @classmethod
    def from_columns(cls, columns: dict[str, list[Any]]) -> "IntermediateResult":
        def run(consumer: Consumer) -> None:
            n = len(next(iter(columns.values())))
            for i in range(n):
                consumer({col: vals[i] for col, vals in columns.items()})
        return cls(run)

    def run(self, consumer: Consumer) -> None:
        self._run(consumer)

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "IntermediateResult":
        def run(consumer: Consumer) -> None:
            self.run(lambda row: consumer(row) if predicate(row[column]) else None)
        return IntermediateResult(run)

    def join(self, other: Query, left_on: str, right_on: str) -> "IntermediateResult":
        def run(consumer: Consumer) -> None:
            right_rows: list[dict[str, Any]] = []
            IntermediateResult.from_columns(other.collect()._columns).run(right_rows.append)
            def joined(left_row: dict[str, Any]) -> None:
                for right_row in right_rows:
                    if left_row[left_on] == right_row[right_on]:
                        consumer(left_row | right_row)
            self.run(joined)
        return IntermediateResult(run)

    def collect(self) -> "DataFrame":
        columns: dict[str, list[Any]] = {}
        def accumulate(row: dict[str, Any]) -> None:
            for col, val in row.items():
                columns.setdefault(col, []).append(val)
        self.run(accumulate)
        return DataFrame(columns)
