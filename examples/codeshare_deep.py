import csv
from pathlib import Path

from micro_dataframes.deep import (
    Filter,
    Join,
    Limit,
    Source,
    collect,
    from_rows,
)


def read_csv(path: Path) -> Source:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return Source(from_rows([dict(row) for row in rows]))


routes = read_csv(Path("examples/openflights/routes.csv"))
airlines = read_csv(Path("examples/openflights/airlines.csv"))

plan = Limit(
    Filter(
        Filter(
            Join(routes, airlines, "route-airline-id", "airline-id"),
            "codeshare", lambda v: v == "Y"),
        "name", lambda v: v == "American Airlines"),
    3)

result = collect(plan)

for row in result["source-airport"]:
    print(row)
