# micro-dataframes

Small dataframe implementations, each around a hundred lines, illustrating
different ways to embed a query DSL in Python. They all run the same query:
join two OpenFlights tables, filter, and take the first few rows. Each module
in `src/micro_dataframes/` is self-contained, and each has a runnable example
in `examples/`.

```
uv run python examples/codeshare_eager.py
```

## Implementations

| Module | Style | What it shows |
|---|---|---|
| `eager.py` | Fluent, eager | The baseline: each method materializes its result immediately. No query representation exists, so nothing can be optimized. |
| `functional.py` | Shallow embedding, functions | A query is just a thunk producing rows. The representation is the public interface, so users can write new operators (see `codeshare_functional_extension.py`) exactly like the built-in ones. |
| `query_lift.py` | Fluent, explicit lift | Separates `DataFrame` (data) from `Query` (computation). The user lifts explicitly with `q(df)` and lowers with `collect()`. |
| `query_forward.py` | Fluent, implicit lift | Same split, but `DataFrame` forwards query methods to `IntermediateResult`, so lifting is invisible at the call site. |
| `lazy_pull.py` | Shallow embedding, pull | `IntermediateResult` wraps a function returning a row iterator. Lazy, streaming, Volcano-style: consumers pull rows on demand. |
| `lazy_push.py` | Shallow embedding, push | The dual: producers push rows into consumer callbacks. `limit` must escape with an exception because a consumer can't stop its producer. |
| `deep.py` | Deep embedding | Queries are `Plan` dataclasses (`Source`, `Filter`, `Join`, `Limit`) walked by an interpreter. The program is data. |
| `deep_pushdown.py` | Deep embedding + optimizer | Adds a `schema` pass and an `optimize` pass that pushes filters below joins. The payoff of having the program as data. |
| `fluent_pushdown.py` | Fluent frontend, deep backend | The fluent interface builds `Plan` nodes instead of executing; `collect()` optimizes then interprets. Roughly how Polars' lazy API is shaped. |
| `pipe_rows.py` | Deep + sanctioned extension point | `fluent_pushdown` plus `pipe_rows(fn)`, an escape hatch taking an opaque iterator transformer. The optimizer must treat it as a barrier. |
| `codegen.py` | Deep embedding + code generation | After pushdown, compiles the plan to one fused Python function (nested loops, inline filters, limit counter) via `compile`/`exec`. Lambdas and source columns are injected into the generated code's environment. |
| `arrow.py` | Eager on Arrow kernels | Executes on `pyarrow` compute kernels. `filter` takes a small expression DSL (`col("x") == "Y"`) instead of a lambda, because the backend can't run opaque Python functions over its columns. |

Two examples extend implementations from the outside:
`codeshare_functional_extension.py` adds a `distinct` operator to the
functional version, and `codeshare_monkeypatch.py` adds one to `lazy_pull` by
assigning a method onto the class at runtime.

## Data

`examples/openflights/` contains route and airline tables from
[OpenFlights](https://openflights.org/data.php). The CSV loader keeps every
value as a string.
