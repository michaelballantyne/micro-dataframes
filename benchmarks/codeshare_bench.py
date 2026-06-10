# Benchmark the codeshare query across implementations.
#
# Query: routes JOIN airlines ON route-airline-id = airline-id
#        WHERE codeshare = 'Y' AND name = 'American Airlines'
#
# Two variants: full materialization (no limit), and limit(3).
#
# eager runs the unpushed join in O(n*m); on the full data (67k x 6k =
# 400M iterations) that takes minutes, so it gets a documented subset.
# The first AA codeshare route appears at index 4655, so the subset is
# chosen to include some matching rows.
#
# Run with: uv run python benchmarks/codeshare_bench.py

import csv
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from micro_dataframes import arrow, codegen, eager, fluent_pushdown, lazy_pull, vectorized
from micro_dataframes.arrow import col

DATA_DIR = Path(__file__).parent.parent / "examples" / "openflights"
EAGER_SUBSET = 6_000

Rows = list[dict[str, Any]]


def load(name: str) -> Rows:
    with (DATA_DIR / f"{name}.csv").open() as f:
        return [dict(row) for row in csv.DictReader(f)]


# The same query against any of the lambda-filter implementations.  eager
# has no collect(); the others need it to run.
def lambda_query(mod: Any, routes: Rows, airlines: Rows, n: int | None) -> Any:
    q = (
        mod.DataFrame(routes)
        .join(mod.DataFrame(airlines), left_on="route-airline-id", right_on="airline-id")
        .filter("codeshare", lambda v: v == "Y")
        .filter("name", lambda v: v == "American Airlines")
    )
    if n is not None:
        q = q.limit(n)
    return q.collect() if hasattr(q, "collect") else q


# arrow filters with the expression DSL and is eager.
def arrow_query(routes: Rows, airlines: Rows, n: int | None) -> Any:
    q = (
        arrow.DataFrame(routes)
        .join(arrow.DataFrame(airlines), left_on="route-airline-id", right_on="airline-id")
        .filter(col("codeshare") == "Y")
        .filter(col("name") == "American Airlines")
    )
    return q if n is None else q.limit(n)


# vectorized uses the expression DSL and is eager (selection-vector model).
def vectorized_query(routes: Rows, airlines: Rows, n: int | None) -> Any:
    q = (
        vectorized.DataFrame(routes)
        .join(vectorized.DataFrame(airlines), left_on="route-airline-id", right_on="airline-id")
        .filter(vectorized.col("codeshare") == "Y")
        .filter(vectorized.col("name") == "American Airlines")
    )
    return q if n is None else q.limit(n)


def best_of_3(fn: Callable[[], Any]) -> tuple[float, Any]:
    best = float("inf")
    result: Any = None
    for _ in range(3):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def bench(n: int | None, routes: Rows, airlines: Rows, routes_sub: Rows) -> None:
    runs: list[tuple[str, Rows, Callable[[Rows], Any]]] = [
        ("eager", routes_sub, lambda r: lambda_query(eager, r, airlines, n)),
        ("lazy_pull", routes, lambda r: lambda_query(lazy_pull, r, airlines, n)),
        ("fluent_pushdown", routes, lambda r: lambda_query(fluent_pushdown, r, airlines, n)),
        ("codegen", routes, lambda r: lambda_query(codegen, r, airlines, n)),
        ("arrow", routes, lambda r: arrow_query(r, airlines, n)),
        ("vectorized", routes, lambda r: vectorized_query(r, airlines, n)),
    ]
    print(f"{'implementation':<16} {'routes rows':>12} {'result rows':>12} {'best of 3':>12}")
    for name, data, fn in runs:
        secs, result = best_of_3(lambda: fn(data))  # noqa: B023
        nrows = len(result["source-airport"])
        print(f"{name:<16} {len(data):>12,} {nrows:>12} {secs:>11.4f}s")


def main() -> None:
    routes = load("routes")
    airlines = load("airlines")
    routes_sub = routes[:EAGER_SUBSET]
    print(f"routes: {len(routes):,} rows   airlines: {len(airlines):,} rows")
    print(f"eager uses the first {EAGER_SUBSET:,} routes; everything else uses all of them.\n")

    print("=== join + two filters, no limit ===")
    bench(None, routes, airlines, routes_sub)

    print("\n=== join + two filters, limit(3) ===")
    bench(3, routes, airlines, routes_sub)

    print(
        "\nThings to notice:\n"
        "  * eager joins before filtering, so it is slow even on a tenth of the data.\n"
        "  * lazy_pull avoids materializing intermediate frames but doesn't reorder\n"
        "    anything; it still filters after the join.\n"
        "  * fluent_pushdown and codegen run the same optimized plan; codegen also\n"
        "    removes the per-row dict machinery by compiling a fused kernel.\n"
        "  * vectorized pays interpreter overhead once per column instead of once\n"
        "    per row.  Its join is a hash join, though, while the pure-Python\n"
        "    family scans the (filtered) right side per left row - it has no\n"
        "    optimizer, so a nested loop over unshrunk inputs would be quadratic.\n"
        "    arrow also hash-joins internally, so those two rows mix a join-\n"
        "    algorithm difference into the execution-model comparison.\n"
        "  * arrow's per-query time is dominated by fixed overhead (building the\n"
        "    pyarrow table from Python dicts); its kernels scale far better\n"
        "    than pure Python as data grows.\n"
        "  * limit(3) helps the streaming implementations and codegen (they stop\n"
        "    early); eager, vectorized, and arrow compute everything regardless."
    )


if __name__ == "__main__":
    main()
