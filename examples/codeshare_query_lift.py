from loader import load_rows

from micro_dataframes.query_lift import DataFrame, q

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))

result = (
    q(routes)
    .join(q(airlines), left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .filter("name", lambda v: v == "American Airlines")
    .limit(3)
    .collect()
)

print(result["source-airport"])
