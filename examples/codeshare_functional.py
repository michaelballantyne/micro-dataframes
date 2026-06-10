from loader import load_rows

from micro_dataframes.functional import collect, filter, from_rows, join, limit, source, thread

routes = source(from_rows(load_rows("routes")))
airlines = source(from_rows(load_rows("airlines")))

# Nested, inside-out.
query = limit(
    filter(
        filter(
            join(routes, airlines, "route-airline-id", "airline-id"),
            "codeshare", lambda v: v == "Y"),
        "name", lambda v: v == "American Airlines"),
    3)

print(collect(query)["source-airport"])

# The same query threaded, top-to-bottom.
query = thread(
    join(routes, airlines, "route-airline-id", "airline-id"),
    lambda q: filter(q, "codeshare", lambda v: v == "Y"),
    lambda q: filter(q, "name", lambda v: v == "American Airlines"),
    lambda q: limit(q, 3),
)

print(collect(query)["source-airport"])
