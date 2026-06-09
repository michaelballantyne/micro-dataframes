import csv
from pathlib import Path
from typing import Any


def load_rows(name: str) -> list[dict[str, Any]]:
    path = Path(__file__).parent / "openflights" / f"{name}.csv"
    with path.open() as f:
        return [dict(row) for row in csv.DictReader(f)]
