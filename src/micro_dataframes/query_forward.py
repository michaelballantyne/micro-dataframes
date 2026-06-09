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


class IntermediateResult(Query):
    _columns: dict[str, list[Any]]

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns

    @classmethod
    def lift(cls, query: Query) -> IntermediateResult:
        if isinstance(query, IntermediateResult):
            return query
        return cls(query.collect()._columns)

    def _nrows(self) -> int:
        return len(next(iter(self._columns.values())))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        mask = [predicate(v) for v in self._columns[column]]
        return IntermediateResult({col: [v for v, keep in zip(vals, mask, strict=True) if keep]
                                   for col, vals in self._columns.items()})

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        other_ir = IntermediateResult.lift(other)
        result: dict[str, list[Any]] = {col: [] for col in self._columns | other_ir._columns}
        for i in range(self._nrows()):
            for j in range(other_ir._nrows()):
                if self._columns[left_on][i] == other_ir._columns[right_on][j]:
                    for col in self._columns:
                        result[col].append(self._columns[col][i])
                    for col in other_ir._columns:
                        result[col].append(other_ir._columns[col][j])
        return IntermediateResult(result)

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult({col: vals[:n] for col, vals in self._columns.items()})

    def collect(self) -> DataFrame:
        return DataFrame(self._columns)
