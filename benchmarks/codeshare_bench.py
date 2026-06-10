# Benchmark the codeshare query across implementations.
#
# Query: routes JOIN airlines ON route-airline-id = airline-id
#        WHERE codeshare = 'Y' AND name = 'American Airlines'
#
# Two variants: full materialization (no limit), and limit(3).
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


def bench(n: int | None, routes: Rows, airlines: Rows) -> None:
    runs: list[tuple[str, Callable[[], Any]]] = [
        ("eager", lambda: lambda_query(eager, routes, airlines, n)),
        ("lazy_pull", lambda: lambda_query(lazy_pull, routes, airlines, n)),
        ("fluent_pushdown", lambda: lambda_query(fluent_pushdown, routes, airlines, n)),
        ("codegen", lambda: lambda_query(codegen, routes, airlines, n)),
        ("arrow", lambda: arrow_query(routes, airlines, n)),
        ("vectorized", lambda: vectorized_query(routes, airlines, n)),
    ]
    print(f"{'implementation':<16} {'routes rows':>12} {'result rows':>12} {'best of 3':>12}")
    for name, fn in runs:
        secs, result = best_of_3(fn)
        nrows = len(result["source-airport"])
        print(f"{name:<16} {len(routes):>12,} {nrows:>12} {secs:>11.4f}s")


def main() -> None:
    routes = load("routes")
    airlines = load("airlines")
    print(f"routes: {len(routes):,} rows   airlines: {len(airlines):,} rows\n")

    print("=== join + two filters, no limit ===")
    bench(None, routes, airlines)

    print("\n=== join + two filters, limit(3) ===")
    bench(3, routes, airlines)

    print(
        "\nThings to notice:\n"
        "  * Every implementation uses the same hash join, so the spread (about\n"
        "    4x) measures materialization strategy and per-row interpretation,\n"
        "    not join algorithm.\n"
        "  * eager is slowest because it materializes the full join result -\n"
        "    every matching row, every column - before the filters see any of it.\n"
        "  * lazy_pull streams rows instead, and limit(3) lets it stop early.\n"
        "  * fluent_pushdown runs the filters before the join, so the build side\n"
        "    shrinks to a single airline.\n"
        "  * codegen runs the same optimized plan with no per-row dict machinery,\n"
        "    making it the fastest pure-Python version.\n"
        "  * vectorized and arrow pay fixed per-query costs (whole-column passes,\n"
        "    pyarrow table construction); their advantage grows with data size,\n"
        "    arrow's especially since its kernels leave Python entirely."
    )


if __name__ == "__main__":
    main()
