# micro-dataframes

Small dataframe implementations, each around a hundred lines, illustrating
different ways to embed a query DSL in Python. They all run the same query:
join two OpenFlights tables, filter, and take the first few rows. Each module
in `src/micro_dataframes/` is self-contained, and each has a runnable example
in `examples/`.

## Simplifying assumptions

All implementations share these, so the differences between files are
embedding and execution style, not features or algorithms:

- The only operations are `filter`, inner `join`, and `limit` — no select,
  group-by, aggregation, or sort.
- Row order is undefined. There is no order-by, so `limit(3)` may return any
  three matching rows.
- Joins are single-key equality joins, and the joined tables are assumed to
  have disjoint column names.
- Every implementation uses the same join algorithm: a hash join that builds
  an index over the right input and probes it with the left.
- Data is small and fully in memory; every row has every column, and there
  are no nulls. The CSV loader keeps every value as a string.
- When errors are reported (e.g. a misspelled column name) is deliberately
  unspecified — it varies by embedding, and
  `tests/test_error_staging.py` records where each one fails.

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
| `deep.py` | Deep embedding | Queries are `Plan` dataclasses (`Source`, `Filter`, `Join`, `Limit`) walked by an interpreter. |
| `deep_pushdown.py` | Deep embedding + optimizer | Adds a `schema` pass and an `optimize` pass that pushes filters below joins — possible only because the query is a data structure. |
| `fluent_pushdown.py` | Fluent frontend, deep backend | The fluent interface builds `Plan` nodes instead of executing; `collect()` optimizes then interprets. Roughly how Polars' lazy API is shaped. |
| `pipe_rows.py` | Deep + sanctioned extension point | `fluent_pushdown` plus `pipe_rows(fn)`, an escape hatch taking an opaque iterator transformer. The optimizer must treat it as a barrier. |
| `codegen.py` | Deep embedding + code generation | After pushdown, compiles the plan to one fused Python function: loops, inline filter guards, a generated hash-join build and probe, a limit counter. The kernel is built as a Python AST with a small quasiquote helper and `compile`/`exec`; lambdas and source columns are injected into its namespace. The example prints the generated kernel. |
| `vectorized.py` | Columnar interpreter, selection vectors | Whole-column kernels are the only place row loops live; operators compose them. `filter` and `limit` shrink a selection vector instead of copying data; `join` is a hash join over index vectors. The predicate DSL is a small deep embedding evaluated to a mask. |
| `arrow.py` | Eager on Arrow kernels | The same execution model as `vectorized.py`, delegated to `pyarrow`'s compute kernels. `filter` takes the expression DSL (`col("x") == "Y"`) instead of a lambda, because the backend can't run opaque Python functions over its columns. |

Two examples extend implementations from the outside:
`codeshare_functional_extension.py` adds a `distinct` operator to the
functional version, and `codeshare_monkeypatch.py` adds one to `lazy_pull` by
assigning a method onto the class at runtime.

## Extras

These are kept separate from the implementations and examples:

- `docs/embeddings.md` — what the implementations show, organized by
  concept: syntax, shallow vs deep, optimizations, extensibility, with
  exercises and further reading. The table above is in a sensible reading
  order.
- `tests/` — differential tests running the same queries through every
  implementation, plus tests pinning down when each embedding reports errors.
- `benchmarks/` — timings comparing no optimization, predicate pushdown,
  the compiled kernel, and the two columnar versions.

## Data

`examples/openflights/` contains route and airline tables from
[OpenFlights](https://openflights.org/data.php). The CSV loader keeps every
value as a string.
