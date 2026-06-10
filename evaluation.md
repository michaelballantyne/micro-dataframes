# Evaluation: this repo as teaching material for DSL embeddings

An assessment of the collection as a vehicle for teaching embedded DSL design,
written after adding the `codegen`, `vectorized`, and `arrow` implementations.

## What works

The core arc is the right one, and each step is small enough to read in a few
minutes:

1. `eager.py` — no embedding at all; the API is just methods that compute.
2. `functional.py` / `lazy_pull.py` / `lazy_push.py` — shallow embeddings;
   a query is a function. Laziness appears, and the pull/push contrast is
   instructive (`limit` is one line in pull and needs an exception in push).
3. `deep.py` / `deep_pushdown.py` — the program becomes data, which makes an
   optimizer possible for the first time.
4. `fluent_pushdown.py` — the user-facing surface and the representation
   decouple: fluent frontend, plan backend. This is the shape of Polars'
   lazy API, Spark DataFrames, and dask.
5. `codegen.py` / `vectorized.py` / `arrow.py` — three more execution
   models: the plan compiled to a fused Python kernel; columnar execution
   with whole-column kernels and selection vectors; and the same columnar
   model delegated to Arrow's compiled kernels.

The extension trio (`codeshare_functional_extension.py`,
`codeshare_monkeypatch.py`, `pipe_rows.py`) is the most distinctive part of
the collection. Most embedding tutorials stop at "deep enables optimization";
these three show the cost — a deep embedding is closed, and every way of
reopening it (a representation that is the interface, monkey-patching, a
sanctioned opaque escape hatch) trades away either static checking or
optimizer freedom. `pipe_rows.py` makes the optimizer's conservatism around
opaque code concrete in five lines of the `optimize` pass.

The most important idea, which the collection now demonstrates end-to-end:
what the system can do is bounded by how much of the user's program it can
see. A lambda predicate is opaque, so the interpreter can only call it; a
`(column, lambda)` pair exposes just enough structure for pushdown; the
`arrow` version's expression DSL exposes the whole predicate, which is what a
vectorized or compiling backend needs. The change in `filter`'s signature
across versions is the visible trace of this.

## What was missing

(The first four items below have since been filled in on this branch, in
`tests/`, `benchmarks/`, and `docs/` — kept separate from the implementations
so they are easy to drop.)

- **Tests.** `tests/` was empty. A differential test running the same queries
  through every implementation and asserting identical results catches
  regressions and states the shared semantics explicitly, which is itself a
  teaching point.
- **Prose.** The rationale for each module lived only in commit messages.
  Notes per module saying what it demonstrates and what to compare it against
  let the files stand alone as course material (now in `docs/tour.md`).
- **Observable optimization.** Pushdown and fusion were claimed, not shown. A
  small benchmark makes `eager` vs `fluent_pushdown` vs `codegen` differ
  measurably, not just architecturally.
- **Error staging.** When does a misspelled column name fail? Immediately in
  `eager`, at `collect()` in the lazy versions, while compiling the plan in
  `codegen`. Error locality is a standard embedding trade-off and the
  material was already here; `tests/test_error_staging.py` now records it.
- **Projection.** A `select` operator would enable projection pushdown, the
  other canonical optimization, and would make the `schema` pass do more work.
  Probably the right cut for size, but it is the most natural extension, and
  it remains open as an exercise.

## Confusing or wrong

- `IntermediateResult` names four different things across modules: a
  materialized table (`query_lift`, `query_forward`), an iterator thunk
  (`lazy_pull`), a consumer-driver (`lazy_push`), and a plan handle
  (`fluent_pushdown`). The uniformity is deliberate and useful for diffing the
  modules, but the name suggests "already computed," which is exactly wrong
  for the lazy versions. `LazyFrame` or `QueryHandle` would mislead less.
- Join behavior on shared column names diverged between the implementations
  before the contract was tightened: the dict-merge joins let the right side
  win, while the column-append joins (`eager`, `query_lift`, `query_forward`)
  collected values from both sides into doubled, ragged columns. The contract
  now assumes disjoint columns, which moves the quirk out of scope rather
  than fixing it — worth a sentence in class either way.
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
actually face in real systems. `codegen`, `vectorized`, and `arrow` are good
capstones: the first shows the plan is a real intermediate representation by
giving it a second backend, and the columnar pair shows the execution model
real engines actually use — once with the kernels visible, once delegated to
a library. `lazy_push` and `query_lift`/`query_forward` are useful contrasts
but could be exercises rather than lecture material.

## Is a dataframe DSL a good choice?

Yes. The domain is familiar, the deep representation (relational plans) is
the textbook one, predicate pushdown is a well-known optimization that fits
in fifteen lines here, and every step has a production system to point at
(pandas is `eager`, Polars' lazy API is `fluent_pushdown`, DuckDB/HyPer are
`codegen`, pandas-on-Arrow is `arrow`). It is hard to find another domain
where laziness, optimization, staging, and extensibility all show up in this
little code.

Two limitations to acknowledge. Column names are strings, so the static-typing
dimension of embeddings (typed schemas, compile-time errors) is mostly out of
reach in Python and goes undemonstrated. And the semantic contract matters
more than it first appears: this repo originally promised exact nested-loop
row order, which quietly constrained the backends — the Arrow join needed
sentinel index columns and a post-join sort to reproduce an order its hash
join doesn't provide. Declaring order undefined (defensible, since there is
no order-by) collapsed that join to a single call. The episode is itself
worth teaching: semantics you don't promise are implementation freedom you
keep.

Alternatives considered: parser combinators (the classic shallow/deep example,
but the optimization story is weaker and the domain less familiar to most
students), build systems, and tensor expressions. Tensor DSLs are the one
serious rival — eager-to-lazy-to-fused-codegen is the same progression as
PyTorch to `torch.compile`, which many students already know, and fusion is
better motivated there. But tensor semantics bring in shapes and
broadcasting, a lot of incidental complexity. Queries are the better trade.

## How the codegen should be done

Three styles were tried for `codegen.py` before settling on one:

1. **String concatenation** (an indentation-tracking emitter, then f-string
   blocks with `textwrap.indent`). Shortest by a wide margin, and the
   produce/consume structure shows clearly because there is little machinery
   around it. But there is no intermediate structure to inspect — mistakes
   (a missing `repr()` around a column name, a dropped indent) surface only
   as syntax errors at `compile()` time.
2. **libcst templates** (`parse_template_statement`). Works, but it is a
   sizable dependency aimed at source rewriting, its templates can't splice
   a multi-statement block into a body slot (every loop needs a follow-up
   `with_changes(body=...)`), and an empty body is a runtime error. The
   friction outweighs what it adds here.
3. **stdlib `ast` with a small homemade quasiquote helper** — templates via
   `ast.parse`, placeholders spliced by a `NodeTransformer`, `ast.unparse`
   for display, `compile()` on the tree directly. This is what was kept. The
   kernel is built as a tree and manipulated as data, which is the same move
   the plan representation makes, so the module teaches one idea twice. Two
   subtleties remain and are worth knowing about: substituted fragments are
   deep-copied so a fragment can appear in several places, and generated
   identifiers are interpolated into the template strings (they are minted
   by the generator, so this is safe) while data and AST fragments go
   through placeholders.

True quasiquoting (mcpyrate) was also considered but requires installing a
macro-expanding import hook, which is too invasive for a repo whose modules
should be plain Python.

## Smaller notes

- The repo's discipline — uniform API, self-contained modules, mypy strict,
  one shared example query — is what makes side-by-side reading work. Resist
  factoring out the duplicated `DataFrame` constructor; the duplication is
  what keeps each file readable alone.
- The example query is well chosen: the join plus two filters exercises
  pushdown on both sides, and `limit` makes laziness matter.
- Consider numbering the modules or adding a suggested reading order; the
  filenames alone don't convey the progression.
