from typing import Any, Callable, Protocol


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
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        return IntermediateResult(self._columns).filter(column, predicate)

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        return IntermediateResult(self._columns).join(other, left_on, right_on)

    # Except collect.
    def collect(self) -> "DataFrame":
        return self


class IntermediateResult(Query):
    _columns: dict[str, list[Any]]

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns

    def _nrows(self) -> int:
        return len(next(iter(self._columns.values())))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "IntermediateResult":
        mask = [predicate(v) for v in self._columns[column]]
        return IntermediateResult({col: [v for v, keep in zip(vals, mask) if keep]
                                   for col, vals in self._columns.items()})

    def join(self, other: Query, left_on: str, right_on: str) -> "IntermediateResult":
        other_ir = other if isinstance(other, IntermediateResult) else IntermediateResult(other.collect()._columns)
        result: dict[str, list[Any]] = {col: [] for col in self._columns | other_ir._columns}
        for i in range(self._nrows()):
            for j in range(other_ir._nrows()):
                if self._columns[left_on][i] == other_ir._columns[right_on][j]:
                    for col in self._columns:
                        result[col].append(self._columns[col][i])
                    for col in other_ir._columns:
                        result[col].append(other_ir._columns[col][j])
        return IntermediateResult(result)

    def collect(self) -> "DataFrame":
        return DataFrame(self._columns)
