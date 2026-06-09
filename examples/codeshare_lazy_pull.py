from loader import load_rows

from micro_dataframes.lazy_pull import DataFrame

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))

result = (
    routes
    .join(airlines, left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .filter("name", lambda v: v == "American Airlines")
    .limit(3)
    .collect()
)

print(result["source-airport"])
