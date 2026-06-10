from collections.abc import Callable
from typing import Any


class DataFrame:
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

    def _nrows(self) -> int:
        return len(next(iter(self._columns.values())))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> DataFrame:
        mask = [predicate(v) for v in self._columns[column]]
        return DataFrame({col: [v for v, keep in zip(vals, mask, strict=True) if keep]
                          for col, vals in self._columns.items()})

    def join(self, other: DataFrame, left_on: str, right_on: str) -> DataFrame:
        # Build: index the right side by key.  Probe: stream the left side.
        index: dict[Any, list[int]] = {}
        for j, key in enumerate(other._columns[right_on]):
            index.setdefault(key, []).append(j)

        result: dict[str, list[Any]] = {col: [] for col in self._columns | other._columns}
        for i in range(self._nrows()):
            for j in index.get(self._columns[left_on][i], []):
                for col in self._columns:
                    result[col].append(self._columns[col][i])
                for col in other._columns:
                    result[col].append(other._columns[col][j])

        return DataFrame(result)

    def limit(self, n: int) -> DataFrame:
        return DataFrame({col: vals[:n] for col, vals in self._columns.items()})

    def __getitem__(self, column: str) -> list[Any]:
        return self._columns[column]
