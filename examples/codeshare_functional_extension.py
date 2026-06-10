from collections.abc import Iterator

from loader import load_rows

from micro_dataframes.functional import (
    Query,
    Row,
    collect,
    filter,
    from_rows,
    join,
    limit,
    source,
)


# A new primitive operator, defined outside the library exactly the way
# the library defines its own. It is not a macro over filter/join/limit:
# it keeps state across rows (the seen set), which no composition of the
# existing operators can express.
def distinct(q: Query, column: str) -> Query:
    def rows() -> Iterator[Row]:
        seen = set()
        for row in q():
            if row[column] not in seen:
                seen.add(row[column])
                yield row
    return rows


routes = source(from_rows(load_rows("routes")))
airlines = source(from_rows(load_rows("airlines")))

# The first three distinct aircraft types American Airlines runs on its
# codeshare routes.
query = limit(
    distinct(
        filter(
            filter(
                join(routes, airlines, "route-airline-id", "airline-id"),
                "codeshare", lambda v: v == "Y"),
            "name", lambda v: v == "American Airlines"),
        "equipment"),
    3)

print(collect(query)["equipment"])
