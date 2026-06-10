from collections.abc import Callable

from loader import load_rows

from micro_dataframes.pipe_rows import DataFrame, Rows, optimize

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))


# An extension at the row-stream level, written without touching the
# library: a generic operator that partially applies to a column,
# yielding the generator transformer pipe_rows expects.
def distinct(column: str) -> Callable[[Rows], Rows]:
    def transform(rows: Rows) -> Rows:
        seen = set()
        for row in rows:
            if row[column] not in seen:
                seen.add(row[column])
                yield row
    return transform


query = (
    routes
    .join(airlines, left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .pipe_rows(distinct("equipment"))
    .filter("name", lambda v: v == "American Airlines")
    .limit(3)
)

print(query.collect()["equipment"])

# The codeshare filter, written before the pipe_rows, is pushed into the
# join. The name filter, written after it, would also like to reach the
# join's right side -- but pipe_rows is opaque to the optimizer (fn may
# be stateful, as distinct is), so the filter is stuck above it.
print(optimize(query._plan))
