from loader import load_rows

from micro_dataframes.codegen import DataFrame

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))

query = (
    routes
    .join(airlines, left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .filter("name", lambda v: v == "American Airlines")
    .limit(3)
)

print("=== generated kernel ===")
print(query.generated_source())
print("========================")

result = query.collect()
print(result["source-airport"])
