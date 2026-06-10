# Evaluation: this repo as teaching material for DSL embeddings

An assessment of the collection as a vehicle for teaching embedded DSL design,
written after adding the `codegen` and `arrow` implementations.

## What works

The core arc is the right one, and each step is small enough to read in a few
minutes:

1. `eager.py` — no embedding at all; the API is just methods that compute.
2. `functional.py` / `lazy_pull.py` / `lazy_push.py` — shallow embeddings;
   a query is a function. Laziness appears, and pull vs push is a genuinely
   instructive duality (`limit` is one line in pull and needs an exception in
   push).
3. `deep.py` / `deep_pushdown.py` — the program becomes data, and the
   immediate payoff is an optimizer that couldn't exist before.
4. `fluent_pushdown.py` — the user-facing surface and the representation
   decouple: fluent frontend, plan backend. This is the shape of Polars,
   Spark, and every modern lazy dataframe.
5. `codegen.py` / `arrow.py` — the plan is retargeted: compiled to a fused
   Python kernel, or executed on vectorized Arrow kernels.

The extension trio (`codeshare_functional_extension.py`,
`codeshare_monkeypatch.py`, `pipe_rows.py`) is the most distinctive part of
the collection. Most embedding tutorials stop at "deep enables optimization";
these three show the cost — a deep embedding is closed, and every way of
reopening it (a representation that is the interface, monkey-patching, a
sanctioned opaque escape hatch) trades away either static checking or
optimizer freedom. `pipe_rows.py` makes the optimizer's conservatism around
opaque code concrete in five lines of the `optimize` pass.

The single most important idea, which the collection now demonstrates
end-to-end, is: **what the system can do is bounded by how much of the user's
program it can see.** A lambda predicate is opaque, so the interpreter can
only call it; a `(column, lambda)` pair exposes just enough structure for
pushdown; the `arrow` version's expression DSL exposes the whole predicate,
which is exactly what a vectorized or compiling backend needs. The `filter`
API changing shape across versions is not an inconsistency — it is the lesson.

## What's missing

- **Tests.** `tests/` is empty. A differential test running the same queries
  through every implementation and asserting identical results would (a) catch
  regressions and (b) state the shared semantics explicitly, which is itself a
  teaching point. This is the highest-value addition.
- **Prose.** Until now the rationale for each module lived only in commit
  messages. A module docstring per file saying what it demonstrates and what
  to compare it against would let the files stand alone as course material.
- **A visible payoff for optimization.** Pushdown and fusion are claimed, not
  shown. A tiny benchmark (or even printing row counts flowing through each
  operator) would make `eager` vs `fluent_pushdown` vs `codegen` differ
  observably, not just architecturally. With `limit(3)` and a pushed filter,
  the numbers are dramatic.
- **Error staging.** When does a typo'd column name fail? Immediately in
  `eager`, at `collect()` in the lazy versions, inside generated code in
  `codegen`. Error locality is a classic embedding trade-off and the material
  is already here; one example would surface it.
- **Projection.** A `select` operator would enable projection pushdown, the
  other canonical optimization, and would make the `schema` pass do more work.
  Probably the right cut for size, but it is the most natural extension.

## Confusing or wrong

- `IntermediateResult` names four different things across modules: a
  materialized table (`query_lift`, `query_forward`), an iterator thunk
  (`lazy_pull`), a consumer-driver (`lazy_push`), and a plan handle
  (`fluent_pushdown`). The uniformity is deliberate and useful for diffing the
  modules, but the name suggests "already computed," which is exactly wrong
  for the lazy versions. `LazyFrame` or `QueryHandle` would mislead less.
- Join semantics are subtle and undocumented: `left_row | right_row` means a
  shared non-key column name silently takes the right side's value, and both
  key columns survive. Fine for the example data, surprising elsewhere.
- The `DataFrame` constructor accepts a list of dicts with missing keys and
  silently builds ragged columns, which later crashes `zip(strict=True)` in
  some versions and not others.
- `query_lift` vs `query_forward` is one idea (explicit vs implicit lifting)
  spread across two near-identical files. Worth keeping both, but they should
  cross-reference each other, or readers will hunt for a bigger difference
  than exists.

## Which versions matter most

If a course has time for four: `eager`, `lazy_pull` (or `functional`),
`deep_pushdown`, `fluent_pushdown`. That covers shallow vs deep, lazy vs
eager, optimization, and the frontend/backend split. `pipe_rows` is the best
fifth — extensibility against an optimizer is the question students will
actually face in real systems. `codegen` and `arrow` are good capstones; they
show the plan is a real intermediate representation by giving it second and
third backends. `lazy_push` and `query_lift`/`query_forward` are useful
contrasts but could be exercises rather than lecture material.

## Is a dataframe DSL a good choice?

Yes — close to optimal for this purpose. The domain is familiar, the deep
representation (relational plans) is the textbook one, predicate pushdown is a
real, famous optimization that fits in fifteen lines here, and every step has
a production system to point at (pandas is `eager`, Polars' lazy API is
`fluent_pushdown`, DuckDB/HyPer are `codegen`, pandas-on-Arrow is `arrow`).
Few domains give you laziness, optimization, staging, and extensibility all
with this little code.

Two limitations to acknowledge. Column names are strings, so the static-typing
dimension of embeddings (typed schemas, compile-time errors) is mostly out of
reach in Python and goes undemonstrated. And the row-ordered semantics
(`limit`) quietly constrains the backends — the Arrow version has to work to
preserve nested-loop join order that a real engine would not promise.

Alternatives considered: parser combinators (the classic shallow/deep example,
but the optimization story is weaker and the domain is less universally
familiar), build systems, and tensor expressions. Tensor DSLs are the one
serious rival — eager-to-lazy-to-fused-codegen is exactly the
PyTorch-to-`torch.compile` story students already live in, and "fusion" is
better motivated there. But tensor semantics drag in shapes and broadcasting,
which would triple the incidental complexity. Queries are the better trade.

## Smaller notes

- The repo's discipline — uniform API, self-contained modules, mypy strict,
  one shared example query — is what makes side-by-side reading work. Resist
  factoring out the duplicated `DataFrame` constructor; the duplication is
  what keeps each file readable alone.
- The example query is well chosen: the join plus two filters exercises
  pushdown on both sides, and `limit` makes laziness matter.
- Consider numbering the modules or adding a suggested reading order; the
  filenames alone don't convey the progression.
