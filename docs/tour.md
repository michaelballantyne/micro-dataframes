# A reading order for the implementations

Each module in `src/micro_dataframes/` is self-contained and around a hundred
lines. They share one query vocabulary (`filter`, `join`, `limit`, `collect`)
and one example query (see `examples/`), so the differences between files
corresponds to differences between embedding styles. This document suggests a
reading order and points out what to look for in each file.
[`embeddings.md`](embeddings.md) covers the same ground organized by concept
instead of by module.

The simplifying assumptions shared by all implementations are listed in the
README; the two that matter most while reading are that row order is
undefined (there is no order-by, so `limit(3)` may return any three rows)
and that every join is the same hash join — build an index over the right
input, probe with the left — so the implementations differ only in
embedding and execution style, never in algorithm.

A recurring question throughout: how much of the user's program can the
library see? Each step makes more of the query visible to the library — first
the sequence of operations, then the plan structure, and finally (in the
Arrow version) the predicates — and each step enables something the previous
one couldn't do.

## 1. `eager.py`

No embedding. `filter` and `join` are ordinary methods that compute
immediately, like pandas. There is no value anywhere that represents the
query, so there is nothing to inspect, reorder, or optimize. The later files
are different ways of fixing that.

If you run `examples/codeshare_eager.py`, note that the join runs on the full
tables before either filter is applied.

## 2. `query_lift.py` and `query_forward.py`

Both split the API in two: `DataFrame` holds data, and a second type
represents a query in progress. In `query_lift.py` the user crosses the
boundary explicitly, with `q(df)` to enter and `collect()` to leave. In
`query_forward.py`, `DataFrame` forwards query methods to
`IntermediateResult`, so entering is implicit and only `collect()` remains
visible.

Execution is still eager in both; only the API changed. The point of the pair
is the boundary itself: explicit lifting marks where the semantics change but
adds noise, implicit lifting is more convenient but hides the boundary. Real
lazy dataframe libraries face the same choice; Polars chose explicit
(`df.lazy()`).

## 3. `functional.py`

A query is a `Callable[[], Iterator[Row]]` — a thunk producing rows — and the
operators are plain functions from query to query. This is a shallow
embedding: the representation is its own meaning, and there is nothing to
inspect.

Because the representation is also the public interface, users can define new
operators outside the library. `codeshare_functional_extension.py` adds a
`distinct` operator that keeps state across rows (a `seen` set), which no
combination of the built-in operators could express. The trade-off is that
the library can never examine a query.

## 4. `lazy_pull.py` and `lazy_push.py`

The same shallow idea behind the fluent API, in two dataflow polarities. In
the pull version, a query wraps a function returning an iterator and the
consumer asks for rows (the Volcano model; Python generators make these
operator definitions nearly identical to `functional.py`). In the push
version, a query wraps a function that sends rows into a consumer callback.

Compare the two `limit` implementations. In pull, the consumer stops asking;
it's a `break`. In push, the consumer has no way to tell the producer to
stop, so the module escapes through a private exception (`_StopPushing`).
This asymmetry is inherent to the two disciplines, not an artifact of this
code; push-based engines handle it with explicit cancellation mechanisms.

## 5. `deep.py`

Queries are now dataclasses: `Source`, `Filter`, `Join`, `Limit`. Building a
query builds a tree, and nothing runs until `execute` interprets it. The
example constructs the plan with bare constructors, which is deliberately
awkward — it shows what the fluent API is sugar for.

The interpreter contains the same code as the operators in `lazy_pull.py`,
moved out of the closures and into one `match` statement. A deep embedding is
a shallow embedding whose function bodies have been turned into data.

## 6. `deep_pushdown.py`

Two new passes over the same plan type: `schema` computes the column set
without executing, and `optimize` pushes a `Filter` below a `Join` when its
column belongs entirely to one side. This is the reason to want a deep
embedding: the library now improves the user's program in a way the user
didn't write.

Note what pushdown requires: it must know which side of the join owns the
filter column, which `schema` answers statically. The predicate itself is
still an opaque lambda, and the optimizer never looks inside it. It can move
the filter only because the column name is part of the plan. This is also why
the Arrow backend later needs a different `filter` API.

## 7. `fluent_pushdown.py`

The method-chaining surface from step 2 now builds plan nodes instead of
executing, and `collect()` runs `optimize` and then the interpreter. This is
the architecture of Polars' lazy API, Spark DataFrames, and dask, reduced to
its skeleton: a fluent surface syntax, a plan representation, optimization
passes, and an execution backend, each replaceable separately.

From here on, the remaining files keep this frontend and change what happens
at `collect()`.

## 8. Extending a closed embedding

The deep embedding got its optimizer by closing the operator set: the `match`
statements in `execute` and `optimize` know exactly four node types. Three
files look at ways to extend it anyway:

- `codeshare_functional_extension.py` (against `functional.py`): the contrast
  case. In the open shallow embedding, extension required nothing special.
- `codeshare_monkeypatch.py` (against `lazy_pull.py`): Python classes are
  open at runtime, so the example assigns a new method onto
  `IntermediateResult`. It works, and the required `# type: ignore` comments
  show the cost: the type checker no longer knows the interface.
- `pipe_rows.py`: the library provides an extension point, `pipe_rows(fn)`,
  which takes an opaque iterator transformer. Extension becomes easy again,
  but look at `optimize`: filters must not move across a `PipeRows` node,
  because `fn` may be stateful. One opaque operator forces the optimizer to
  be conservative around it. Real systems live with this same tension between
  user extensibility and optimizer freedom.

## 9. `codegen.py`

Same frontend, same optimizer; the interpreter is replaced by a compiler.
`collect()` walks the optimized plan and builds a single Python function — a
fused kernel containing all the loops, the filters as inline guards, and the
limit as a counter — then compiles and runs it. The example prints the
generated kernel, which is worth reading alongside the module.

Two parts deserve attention. The first is the produce/consume pattern (from
Neumann's HyPer compiler): each plan node generates its loop or guard and
delegates the loop body to its parent through a callback, so each pipeline
becomes one loop nest with no intermediate row representation. A join breaks
the pipeline: the kernel first runs the right side into column buffers,
indexes them by key, and then probes the index from the main loop nest —
all of it generated code. The second is
the mechanics of generating code at runtime: the kernel is built as a Python
`ast` tree — a deep embedding of Python itself, manipulated as data like the
plan — using a small quasiquote helper (`ast.parse` for templates, a
`NodeTransformer` for substitution). Values that can't appear as literals in
source code, namely the column lists and the predicate lambdas, are injected
into the namespace the kernel is exec'd in and referenced by generated names.

There is also a limitation to notice: since the lambda predicates are opaque,
the generated code can only call them (`pred0(...)`). The kernel fuses the
plumbing but not the predicates.

## 10. `vectorized.py`

A different answer to interpreter overhead than `codegen.py`: instead of
removing the per-row dispatch by compiling it away, amortize it by processing
a whole column per call. The kernels section is the point of the module —
`compare`, the mask combinators, `take`, and `hash_join` are the only places
row loops live; the operators just compose them. `filter` and `limit` never
copy data; they shrink a selection vector (the list of live row indices).
This is the execution model of DuckDB and Polars, reduced to about a hundred
lines.

Since every implementation now uses the same hash join, it is fair to ask
what still separates this from `eager.py` — which was never row-at-a-time
either (its `filter` already computes a whole-column mask). Three things.
First, the predicate: eager calls an opaque lambda once per value, while
here the expression DSL maps onto kernels whose per-element work is just
the comparison — no Python call inside the loop. Second, materialization:
every eager operator copies its full result, while `filter` and `limit`
here never touch the data — they shrink the selection vector, and the
columns are shared between frames; only `join` and `__getitem__`
materialize. Third, organization: eager's loops are scattered through its
operators, while here they live only in the kernels, and the operators are
composition and bookkeeping. The second point is most of the benchmark gap
between the two; the third is what makes the DSL necessary, since an
opaque lambda has nowhere to go when the operators don't loop.

Two things to compare against the rest of the collection. First, lambdas are
gone again: a per-row predicate would defeat whole-column execution, so
`filter` takes the `col("x") == "Y"` expression DSL. Here the DSL is a small
deep embedding — frozen dataclass nodes plus an `eval_pred` interpreter that
maps each node to a kernel call — the same move `deep.py` made for query
plans, now applied to predicates. Second, note what the model gives up:
`limit` can't stop the join early, because each operator runs to completion
over its whole input before the next starts.

Worth knowing: push vs pull (step 4) and row-at-a-time vs column-at-a-time
are different axes. Push and pull are about who drives when results stream
through operators one unit at a time; here each operator finishes before the
next begins, so there is nothing to stream and the distinction collapses
into plain sequencing. Production engines sit in the middle — batches of a
few thousand rows — where both axes are in play at once.

## 11. `arrow.py`

The same columnar model, delegated to a real library: pyarrow's compute
kernels. Each operator is one call into Arrow; nothing about the execution
is visible from here, which is precisely the trade — after `vectorized.py`,
read this as what the model looks like from the outside.

The delegation forces the same API change for a harder reason: a lambda
can't be handed to Arrow at all, since the kernels are compiled C++. The
`Expr` wrapper here is a shallow embedding — it wraps a live
`pyarrow.compute.Expression` and delegates operators to it, where
`vectorized.py` built inspectable nodes. Overloading `==` to return a
non-boolean is standard practice in embedded query DSLs (pandas, SQLAlchemy,
Polars, ORMs), and it has a standard cost: `__eq__`'s contract is violated,
and mypy needs a `type: ignore[override]`.

The `join` method shows what the contract buys. With row order undefined and
column names disjoint, the join is a single delegated call. Under the
stricter contract this repo started with — exact nested-loop row order, both
key columns, right side winning name collisions — the same method needed
sentinel row-index columns, a sort after the join, and rename/drop
bookkeeping, because Arrow's hash join promises none of that. Relaxing the
contract deleted all of it: semantics you don't promise are implementation
freedom you keep.

## Related material

- `tests/` runs the same queries through every implementation and asserts
  they agree, which states the shared semantics executably.
  `test_error_staging.py` records when each embedding reports an error: a
  misspelled column name fails immediately in `eager`, at `collect()` in the
  lazy versions, and when the filter is applied in `arrow`.
- `benchmarks/` measures the differences: pushdown versus none, interpreted
  versus compiled, scalar versus vectorized.
- Possible exercises: add a `select` operator and projection pushdown; make
  the optimizer choose which join input to build the index over (it needs
  cardinality estimates — a new pass over the plan); extend `arrow.py`'s
  `Expr` with arithmetic; try writing `distinct` for `fluent_pushdown.py`
  and see how the closed plan type gets in the way.
