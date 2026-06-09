from loader import load_rows

from micro_dataframes.deep import Filter, Join, Limit, Source, collect, from_rows

routes = Source(from_rows(load_rows("routes")))
airlines = Source(from_rows(load_rows("airlines")))

query = Limit(
    Filter(
        Filter(
            Join(routes, airlines, "route-airline-id", "airline-id"),
            "codeshare", lambda v: v == "Y"),
        "name", lambda v: v == "American Airlines"),
    3)

result = collect(query)

print(result["source-airport"])
