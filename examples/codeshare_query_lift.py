import csv
from pathlib import Path

from micro_dataframes.query_lift import DataFrame, q


def read_csv(path: Path) -> DataFrame:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return DataFrame([dict(row) for row in rows])


routes = read_csv(Path("examples/openflights/routes.csv"))
airlines = read_csv(Path("examples/openflights/airlines.csv"))

result = (
    q(routes)
    .join(q(airlines), left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .filter("name", lambda v: v == "American Airlines")
    .collect()
)

for row in result["source-airport"][:3]:
    print(row)
