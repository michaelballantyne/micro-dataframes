from loader import load_rows

from micro_dataframes.vectorized import DataFrame, col

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))

result = (
    routes
    .join(airlines, left_on="route-airline-id", right_on="airline-id")
    .filter(col("codeshare") == "Y")
    .filter(col("name") == "American Airlines")
    .limit(3)
)

print(result["source-airport"])
