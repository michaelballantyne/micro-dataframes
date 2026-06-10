# micro-dataframes

> [!WARNING]
> Written with lots of help from Claude. I've reviewed most of the implementations in some depth, but less so the last three (codegen, vectorized, arrow) and claude-writeup.md is mostly Claude's work from my outline.

Thirteen small implementations of the same dataframe API (`filter`, inner
`join`, `limit`), around a hundred lines each. They differ in how the query
language is embedded in Python. All of them run this query:

```python
routes.join(airlines, left_on="route-airline-id", right_on="airline-id")
      .filter("codeshare", lambda v: v == "Y")
      .filter("name", lambda v: v == "American Airlines")
      .limit(3)
```

The implementations differ in what that expression evaluates to: a finished
table, a pipeline of generators, a plan tree that gets optimized and
interpreted, a generated Python kernel, or calls into a columnar backend.
The API, the semantics, and even the join algorithm are the same everywhere,
so comparing two files isolates a single design decision.

The code is written to be read, and may be useful if you want to understand:

- how lazy dataframe libraries (Polars, Spark, dask) are structured, and
  why pandas can't reorder your query but Polars can;
- how query engines interpret, compile, or vectorize plans;
- how to design an embedded DSL in Python, and what each choice trades away.

## Getting started

Each module in `src/micro_dataframes/` is self-contained and has a runnable
example in `examples/`:

```
uv run python examples/codeshare_eager.py
```

A short path through the code: `eager.py` (no query representation at all),
then `lazy_pull.py` (queries become streaming computations), then
`deep_pushdown.py` (queries become data, and an optimizer appears), then
`fluent_pushdown.py` (the user-facing surface and the representation
decouple — the architecture of real lazy dataframe libraries). After that,
follow whatever interests you: the implementation table below is in a
sensible reading order, and [`claude-writeup.md`](claude-writeup.md) works
through the ideas in depth, with exercises and further reading.

`tests/` runs the same queries through every implementation and asserts they
agree; `benchmarks/codeshare_bench.py` measures what the design differences
cost on the full data.

## The dimensions

The write-up covers four, briefly:

- **Syntax.** Nested function calls, raw constructors, fluent method
  chains, operator overloading. Independent of everything below: the same
  fluent surface appears here over an eager backend, a lazy one, and a
  plan-building one.
- **What a query expression denotes.** Its result (`eager`), a computation
  to run later (`lazy_pull` and `lazy_push`, which also contrast pull- and
  push-driven streaming), or a data structure describing the query (`deep`
  and everything after). This is the shallow vs deep embedding distinction.
- **What visibility buys.** Each capability needs to see a certain amount
  of the user's program: streaming alone removes intermediate tables and
  enables early exit; a plan tree enables predicate pushdown; holding the
  whole plan enables compiling it to a fused kernel; making predicates data
  (instead of opaque lambdas) enables vectorized execution.
- **Extensibility.** Adding a new operator is free in the open
  function-based embedding, needs monkey-patching or an escape hatch behind
  a fluent class, and undermines the optimizer in a deep one — a small
  instance of the expression problem.

## Implementations

| Module | Style | What it shows |
|---|---|---|
| `eager.py` | Fluent, eager | The baseline: each method materializes its result immediately. No query representation exists, so nothing can be optimized. |
| `functional.py` | FP shallow embedding, functions | A query is just a thunk producing rows. The representation is the public interface, so users can write new operators (see `codeshare_functional_extension.py`) exactly like the built-in ones. |
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

## Data

`examples/openflights/` contains route and airline tables from
[OpenFlights](https://openflights.org/data.php). The CSV loader keeps every
value as a string.
