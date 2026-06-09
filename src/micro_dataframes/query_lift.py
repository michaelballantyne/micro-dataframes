from typing import Any, Callable

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

    def __getitem__(self, column: str) -> list[Any]:
        return self._columns[column]


class Query:
    _columns: dict[str, list[Any]]

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns

    def _nrows(self) -> int:
        return len(next(iter(self._columns.values())))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "Query":
        mask = [predicate(v) for v in self._columns[column]]
        return Query({col: [v for v, keep in zip(vals, mask) if keep]
                      for col, vals in self._columns.items()})

    def join(self, other: "Query", left_on: str, right_on: str) -> "Query":
        result: dict[str, list[Any]] = {col: [] for col in self._columns | other._columns}
        for i in range(self._nrows()):
            for j in range(other._nrows()):
                if self._columns[left_on][i] == other._columns[right_on][j]:
                    for col in self._columns:
                        result[col].append(self._columns[col][i])
                    for col in other._columns:
                        result[col].append(other._columns[col][j])
        return Query(result)

    def collect(self) -> "DataFrame":
        return DataFrame(self._columns)


def q(df: "DataFrame") -> Query:
    return Query(df._columns)



