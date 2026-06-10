from collections.abc import Iterator
from typing import Any

from loader import load_rows

from micro_dataframes.lazy_pull import DataFrame, IntermediateResult


# Python classes are open at runtime, so the fluent interface can be
# extended after the fact: define a method and assign it onto the class.
# The type checker can't see methods added this way -- note the ignores
# here and at the call site below. Abstraction by convention, not
# enforcement.
def distinct(self: IntermediateResult, column: str) -> IntermediateResult:
    def make_iter() -> Iterator[dict[str, Any]]:
        seen = set()
        for row in self._iter():
            if row[column] not in seen:
                seen.add(row[column])
                yield row
    return IntermediateResult(make_iter)


IntermediateResult.distinct = distinct  # type: ignore[attr-defined]

routes = DataFrame(load_rows("routes"))
airlines = DataFrame(load_rows("airlines"))

result = (
    routes
    .join(airlines, left_on="route-airline-id", right_on="airline-id")
    .filter("codeshare", lambda v: v == "Y")
    .filter("name", lambda v: v == "American Airlines")
    .distinct("equipment")  # type: ignore[attr-defined]
    .limit(3)
    .collect()
)

print(result["equipment"])
