# Embedding a query language: what this repo shows

This is the concept-organized companion to [`tour.md`](tour.md), which walks
the modules in reading order. Here the organization is by idea instead:
syntax, shallow versus deep embeddings, the optimizations each makes
possible, and extensibility. Everything is illustrated by one running query
over two OpenFlights tables:

```python
routes.join(airlines, left_on="route-airline-id", right_on="airline-id")
      .filter("codeshare", lambda v: v == "Y")
      .filter("name", lambda v: v == "American Airlines")
      .limit(3)
```

Find the codeshare routes American Airlines flies and take three of them.
Thirteen implementations in `src/micro_dataframes/` run this query with the
same meaning (under the simplifying assumptions listed in the README). What
varies is how the query exists inside Python — and that is the whole
subject.

## 1. Syntax

There are three ways to spell a query in this collection, and the first
thing to notice is that the spelling is independent of everything else.

**Function calls.** In `functional.py`, operators are plain functions from
query to query, and the query above is nested applications:

```python
limit(filter(filter(join(routes, airlines, ...), "codeshare", ...), "name", ...), 3)
```

It reads inside-out, which is honest but backwards from the order the
operations happen in.

**Data constructors.** In `deep.py`'s example, the query is built from the
plan dataclasses directly: `Limit(Filter(Filter(Join(...), ...), ...), 3)`.
This looks almost identical to the function-call spelling, but it is the one
syntax option that cannot hide what is underneath: constructors *are* the
representation. The other two spellings can sit on top of anything.

**Fluent method chains.** Most of the modules use the pandas-style chain
shown at the top. The same fluent surface appears over an eager backend
(`query_forward.py`), a lazy one (`lazy_pull.py`), and a plan-building one
(`fluent_pushdown.py`) — proof that surface syntax and internal
representation are separate decisions. Reading order matches execution
order, which is most of why real libraries choose it.

Two further syntax decisions show up once you commit to fluent chains.

*Where is the boundary between data and query?* `query_lift.py` makes the
user cross it explicitly — `q(df)` to start querying, `collect()` to get a
table back — while `query_forward.py` forwards query methods from
`DataFrame` so the entry is invisible and only `collect()` remains. The two
modules are otherwise identical; they exist to show this one choice.
Explicit lifting marks the place where the semantics changes from "this is a
table" to "this is a pending computation," at the price of noise. Real
libraries split on it: Polars chose explicit (`df.lazy()`), pandas-like APIs
chose invisible.

*How do predicates get in?* Most versions take a column name and a Python
lambda. The columnar versions (`vectorized.py`, `arrow.py`) instead overload
operators so that `col("codeshare") == "Y"` builds a value describing the
comparison. Operator overloading is the third syntax tool, and it has a
known cost: `__eq__` is supposed to return `bool`, and the type checker has
to be told to look away. Why those two versions need this is a semantics
question, not a syntax one — see section 3.5.

## 2. Shallow versus deep

The useful question for classifying every module is: **what host-language
value does a query expression denote?**

### 2.1 The simplest shallow embedding: eager computation

In `eager.py`, a query expression denotes *its own result*. Each method
computes immediately and returns a finished table. There is no
representation of the query anywhere — after `.filter(...)` runs, nothing
remembers that a filter happened.

This is still an embedding: the domain's vocabulary (filter, join, limit)
has been mapped onto host-language values. It is just that the meaning
assigned to each expression is the answer itself. The consequence is total:
nothing can be inspected, reordered, fused, or stopped early, because by the
time you could act, the work is done.

### 2.2 Delayed shallow embeddings: denote the computation

The next move is to make a query denote *a computation that will produce the
result*, while still building that computation by directly composing host
functions — no description of the query exists, only runnable code.

`functional.py` is the purest form: a query is a thunk,
`Callable[[], Iterator[Row]]`, and each operator wraps the one below it.
`lazy_pull.py` is the same idea behind the fluent surface. Rows now *stream*:
a row is produced, flows through every operator, and is either kept or
dropped, with no intermediate table ever materialized.

A streamed computation needs a driver, and there are two choices. In the
**pull** version (`lazy_pull.py`), the consumer asks for rows: each operator
is a generator pulling from the one below. In the **push** version
(`lazy_push.py`), the producer calls a consumer callback for each row. The
two are mirror images, and the place to see that the mirror is imperfect is
`limit`. Pulling, the consumer just stops asking:

```python
for i, row in enumerate(self._iter()):
    if i == n:
        break
    yield row
```

Pushing, the consumer has no way to tell the producer to stop, so the module
escapes through a private exception (`_StopPushing`). That asymmetry is
inherent to the disciplines, not an accident of this code.

Both versions are still shallow: composition happens by calling functions,
and once composed, the pipeline is as opaque as `eager`'s results. You can
run it; you cannot look at it.

### 2.3 Deep embeddings: denote a description

In `deep.py`, a query denotes a *data structure describing the query*:

```python
type Plan = Source | Filter | Join | Limit
```

Building a query allocates a tree. Nothing runs until an interpreter walks
it:

```python
def execute(plan: Plan) -> Iterator[dict[str, Any]]:
    match plan:
        case Filter(child, column, predicate):
            for row in execute(child):
                if predicate(row[column]):
                    yield row
        ...
```

Compare this interpreter with `lazy_pull.py` and you will find the same code,
relocated: the bodies that lived inside each operator's closure now live
inside one `match`, with the closure's captured variables turned into fields
of a node. This transformation — replacing functions by data plus a single
function that interprets the data — is called *defunctionalization*, and it
is the precise sense in which a deep embedding is a shallow one turned
inside out.

What was bought? The program is now a value. `deep_pushdown.py` adds two
more functions over the same `Plan` type: `schema`, which computes the
column set without executing anything, and `optimize`, which rewrites the
tree. Later, `codegen.py` adds a fourth: a compiler. One representation,
many interpretations — execution is now just one thing you can do with a
query, and the others are where all the interesting behavior in sections
3.3–3.4 comes from. (This doc only notes here that *compiling* the
description is an option alongside interpreting it; section 3.4 takes it
seriously.)

`fluent_pushdown.py` completes the architecture: the fluent methods build
plan nodes, and `collect()` optimizes and then interprets. Frontend syntax,
plan representation, transformation passes, execution backend — each
replaceable separately. This is the shape of every modern lazy dataframe
library.

### 2.4 Depth is per-construct, not global

It would be a mistake to call any of these versions simply "deep." In
`fluent_pushdown.py` the *operators* are deep — `Filter` is a node — but the
*predicate* inside the filter is still an opaque lambda. The plan records
that a filter exists and which column it reads; what the test actually does
is invisible.

Each construct of the language gets its own depth decision, and you can see
the dial move within this one repo: column references go from strings inside
opaque calls to plan-visible fields (enough for section 3.3), and predicates
finally become data in `vectorized.py`, whose `col("x") == "Y"` builds a
little tree of `Compare`/`And`/`Or` nodes — a deep embedding of predicates,
nested inside what is otherwise an eager dataframe. How much of the user's
program the library can see is set construct by construct, and every
capability in section 3 has a minimum visibility requirement.

### 2.5 A side effect of depth: when errors happen

Delaying computation delays mistakes. Filter on a misspelled column name
and `eager` raises `KeyError` immediately, at the `.filter(...)` call. The
lazy and deep versions raise nothing until `collect()` — the bad name sits
harmlessly inside a closure or a node until something finally reads it.
`codegen` fails at `collect()` too, but during plan compilation rather than
execution. `arrow` is immediate again, because pyarrow validates expressions
against the schema as filters are applied.

None of these is wrong; the contract deliberately leaves error timing
unspecified. But users experience error locality directly, and it is worth
knowing that choosing an embedding style chooses it for them.
`tests/test_error_staging.py` records where each version fails.

## 3. Optimizations

A useful way to organize what follows: each optimization has a minimum
amount of the program it needs to see. The first two need only *delayed*
computation — shallow is enough. The third needs the operator tree. The
fourth needs to hold the whole plan at once. The fifth needs the predicates
themselves. The benchmark (`benchmarks/codeshare_bench.py`) holds everything
else constant — same data, same query, and deliberately the same hash join
everywhere — so the differences below are attributable.

### 3.1 Deforestation: no intermediate tables

Run the query with no limit, and `eager` takes roughly 0.29s where
`lazy_pull` takes 0.19s. Same join, no optimizer in either. The difference
is that `eager` materializes a complete table after every operator: the join
writes all ~67,000 matched rows times 17 columns into fresh lists, then the
first filter reads every one of those values back and writes the survivors
into another full table, and so on. Each operator boundary is a round trip
through memory. In `lazy_pull`, a row is built once, flows through
join–filter–filter in one pass, and dies at the first predicate it fails;
only the 1,089 survivors are ever written anywhere.

Eliminating the intermediate structures between composed operations is
called *deforestation*. Notice the visibility requirement: none. The
streaming versions get it purely from composing generators — no
representation, no analysis. It is the cheapest optimization in the repo and
the first reason laziness pays even when every row must be produced.

### 3.2 Early termination

Add `limit(3)` back and the streaming versions improve again (`lazy_pull`
drops to ~0.09s): once three rows have survived to the end, the consumer
stops pulling and the sources never finish scanning. `eager` computes
everything and then slices.

Again no representation is needed — but the *dataflow discipline* matters:
pull gets early termination in one `break`, push needs the exception escape.
This is the first place the implementations differ in what they can do, not
just how they read.

### 3.3 Predicate pushdown: rewriting the program

The query filters *after* joining, which is the natural way to write it but
a wasteful way to run it: the join builds its index from all 6,162 airlines
when only one survives the name filter. No shallow version can fix this —
fixing it means *reordering the user's program*, and there is no program to
reorder, only composed closures.

With the plan as data, the fix is a pattern match (`deep_pushdown.py`):

```python
case Filter(child, column, predicate):
    match optimize(child):
        case Join(left, right, left_on, right_on) if column in schema(left):
            return Join(optimize(Filter(left, column, predicate)),
                        right, left_on, right_on)
        case Join(left, right, left_on, right_on) if column in schema(right):
            return Join(left,
                        optimize(Filter(right, column, predicate)),
                        left_on, right_on)
```

A filter slides below a join when its column belongs entirely to one side —
a fact `schema` computes statically, without running anything. After
pushdown, the join's build side is one airline instead of 6,162, and the
benchmark shows it: `fluent_pushdown` runs the no-limit query in ~0.13s
against `lazy_pull`'s 0.19s, and the gap widens with data size.

Note the exact visibility requirement: the optimizer never looks inside the
predicate — it can't, the lambda is opaque — but it doesn't need to. It
needs the operator tree and the column *name*, which the filter declares in
plan-visible position. Minimum disclosure, minimum capability. Also note
that this is the first optimization that can be *unsound* if done carelessly:
`pipe_rows.py` (section 4) shows what happens when an operator the optimizer
cannot understand enters the tree.

### 3.4 Removing interpretive overhead: compile the plan

The interpreter pays a tax on every row: dict-rows are allocated and merged,
and the `match` in `execute` re-dispatches on node types constantly. The
plan doesn't change between rows, so this work is redundant — and because
the deep embedding holds the whole program as data, it can be paid once
instead, by *generating code*.

`codegen.py` keeps the frontend and optimizer of `fluent_pushdown.py` and
replaces the interpreter with a compiler. `collect()` walks the optimized
plan and builds a single Python function in which all the per-row decisions
have been made ahead of time. For the running query, the generated kernel
looks like this (abbreviated):

```python
def kernel():
    out = {'active': [], 'airline': [], ...}
    _lim0 = 0
    _right0 = {'airline-id': [], 'name': [], ...}
    def _fill0():
        for _i0 in range(len(source0['airline-id'])):
            if pred0(source0['name'][_i0]):
                _right0['airline-id'].append(source0['airline-id'][_i0])
                ...
    _fill0()
    _idx0 = {}
    for _i1 in range(len(_right0['airline-id'])):
        _idx0.setdefault(_right0['airline-id'][_i1], []).append(_i1)
    for _i3 in range(len(source1['airline'])):
        if pred1(source1['codeshare'][_i3]):
            for _i2 in _idx0.get(source1['route-airline-id'][_i3], []):
                if _lim0 == 3:
                    return out
                _lim0 += 1
                out['source-airport'].append(source1['source-airport'][_i3])
                ...
    return out
```

Everything from sections 3.1–3.3 is visible in one place: no intermediate
tables (values go straight from source columns to output columns), the limit
as a counter with an early return, and the pushed-down filters as inline
guards (`pred1` runs during the join's build phase). The join builds and
probes a dict index inside the kernel; a limit somewhere in the build side
only ends `_fill0`, not the whole kernel.

Three mechanics deserve attention, because they generalize beyond this repo:

- *The kernel is built as a tree, not as text.* The generator constructs
  Python `ast` nodes — a deep embedding of Python itself, manipulated as
  data exactly like the plan — using a ~30-line quasiquote helper:
  `ast.parse` turns template strings into trees, and a placeholder `Name`
  inside a template is replaced by an already-built fragment.
  `ast.unparse` renders the result for inspection (the example prints it);
  `compile` executes the tree directly.
- *Runtime values are injected by name.* The source columns and the
  predicate lambdas cannot appear as literals in generated code, so they are
  placed in the namespace the kernel is executed in, under generated names
  (`source0`, `pred0`), and the code refers to them symbolically. This trick
  — generated code closing over live values — is what makes runtime code
  generation practical.
- *Each operator generates its outer structure and delegates its body.* The
  generator recurses over the plan with a callback (`produce`/`consume`):
  a filter contributes an `if` and asks its parent what goes inside; a
  source contributes a loop. The pipeline collapses into one loop nest with
  no per-row representation at all — fusion, performed at code level.

What remains in the kernel is the irreducible work plus one tax the compiler
cannot remove: `pred0(...)` is still an opaque Python call per row, because
the predicates were never made visible. The benchmark: ~0.08s, the fastest
pure-Python version, with the layers it peeled measurable along the way
(per-row dicts and dispatch from `lazy_pull`'s 0.19s, the remaining
interpretation from `fluent_pushdown`'s 0.13s).

### 3.5 Vectorized operations: amortize instead of eliminate

There is a second answer to interpretive overhead: keep the interpreter, but
make each interpreted step do a *column* of work instead of a row of work,
so the dispatch cost is divided by the number of rows.

`vectorized.py` shows the model with everything visible. A handful of
kernels are the only places row loops exist —

```python
def compare(left, cmp_op, right):
    return [cmp_op(v, right) for v in left]
```

— and the operators just compose them. `filter` evaluates the predicate
over whole columns to a boolean mask and then shrinks a *selection vector*
(the list of live row indices) without copying any data; `limit` slices the
selection vector; only `join` and column access materialize.

This is why the predicate DSL is forced here, completing the thread from
sections 1 and 2.4: a per-row lambda would put an opaque Python call back
inside the kernel loop, defeating the amortization. The predicate must
arrive as data — `col("codeshare") == "Y"` builds `Compare`/`And`/`Or`
nodes — so that an evaluator can map each node to one whole-column kernel
call. To vectorize, the embedding of predicates must go deep.

In pure Python the amortization is real but modest (~0.15s; the per-element
comprehension step is itself interpreted). The model's value shows fully
when the kernels leave Python: `arrow.py` is the same architecture with the
kernels supplied by pyarrow's compiled C++, every operator a single library
call. Same per-query time at this data size — fixed costs dominate — but
the kernels scale far better as data grows. Reading `vectorized.py` first
makes `arrow.py` legible: it is the same model viewed from outside.

One honest note: `vectorized` and `arrow` cannot stop early — every operator
runs to completion over its whole input, so `limit` saves nothing. Each
strategy in this section gives something up; there is no point on the menu
that dominates.

## 4. Extensibility

A user wants `distinct(column)` — keep the first row for each value of a
column. It cannot be built from filter/join/limit (it needs memory across
rows), so it must be a new operator. How hard that is decomposes into three
separate questions, and each embedding style fails a different one.

**Can you extend the computation?** In the function-based shallow embedding,
trivially: the representation (`Callable[[], Iterator[Row]]`) is the public
interface, so `codeshare_functional_extension.py` defines `distinct` outside
the library, with exactly the same standing as the built-in operators —
including its private `seen` set. Behind a fluent surface, the same power
can be offered as a sanctioned escape hatch taking an iterator transformer;
on a shallow representation it is three lines, since transforming the row
stream is already what every operator does:

```python
def pipe_rows(self, fn: Callable[[Rows], Rows]) -> IntermediateResult:
    def make_iter() -> Iterator[dict[str, Any]]:
        yield from fn(self._iter())
    return IntermediateResult(make_iter)
```

On a deep representation (`pipe_rows.py`) the same idea must be reified as a
plan node and handled by every pass. So computation-extension is available
at every depth; depth only adds ceremony.

**Can you extend the syntax?** This is where fluent surfaces resist,
independent of depth. `.pipe_rows(distinct("equipment"))` works but is
second-class next to `.distinct("equipment")`; making the extension a real
method means getting inside the class. Python's classes are open at runtime,
so `codeshare_monkeypatch.py` simply assigns a method onto
`IntermediateResult` — it works, and the required `# type: ignore` comments
itemize the cost: the type checker no longer knows the interface. The
function-based surface has no syntax/representation gap, so its extensions
are first-class automatically. Fluent syntax trades this away for
readability.

**What does the system's reasoning lose?** Nothing — until there are
passes. The shallow versions have no optimizer, so an opaque extension costs
nothing. In `pipe_rows.py`, the optimizer must be told what it cannot know:

```python
# A filter never moves below a PipeRows: fn is opaque and may be
# stateful (e.g. distinct), so reordering would be unsound.
```

One unanalyzable operator in the tree, and pushdown must stop at it forever.
This tension — users want to extend, optimizers want to understand — is
permanent and unresolved in production systems too.

There is a classical name for the underlying trade: the *expression
problem*. The shallow embedding makes new operators easy and new
*interpretations* of existing queries impossible (you cannot add `schema` to
a closure). The deep embedding supports four interpretations of one plan
type in this repo (`execute`, `schema`, `optimize`, the compiler) but makes
a new operator touch all of them. Neither side wins; you choose which axis
of growth to make cheap.

## 5. A note on semantics contracts

One lesson here was learned during construction rather than designed in.
Early versions of this repo promised that joins preserve nested-loop row
order. The pure-Python implementations provide that order for free, so the
promise looked free — until `arrow.py`, where reproducing it on top of a
hash join required sentinel index columns, a post-join sort, and cleanup.
When the contract was relaxed (no order-by means order is undefined), all of
that machinery deleted itself and the join became a single delegated call.

The general point: every guarantee in a DSL's contract constrains all
present and future backends, including the ones you haven't imagined.
Semantics you don't promise are implementation freedom you keep.

## Map

| Section | Modules |
|---|---|
| 1 Syntax | `functional`, `deep` (constructors), `query_lift` / `query_forward`, any fluent module, `vectorized` / `arrow` (overloading) |
| 2.1 Eager shallow | `eager` |
| 2.2 Delayed shallow | `functional`, `lazy_pull`, `lazy_push` |
| 2.3 Deep | `deep`, `deep_pushdown`, `fluent_pushdown` |
| 2.4 Per-construct depth | `fluent_pushdown` vs `vectorized` predicates |
| 2.5 Error staging | `tests/test_error_staging.py` |
| 3.1–3.2 Deforestation, early exit | `lazy_pull`, `lazy_push` vs `eager` |
| 3.3 Pushdown | `deep_pushdown`, `fluent_pushdown` |
| 3.4 Compilation | `codegen` |
| 3.5 Vectorization | `vectorized`, `arrow` |
| 4 Extensibility | `codeshare_functional_extension.py`, `codeshare_monkeypatch.py`, `pipe_rows` |
| 5 Contracts | `arrow`, git history |

## Further reading

- **Shallow and deep embeddings.** Jeremy Gibbons and Nicolas Wu, *Folding
  Domain-Specific Languages: Deep and Shallow Embeddings* (ICFP 2014). The
  clearest formal account of the correspondence sketched in section 2.
- **Defunctionalization.** John Reynolds, *Definitional Interpreters for
  Higher-Order Programming Languages* (1972); Olivier Danvy and Lasse
  Nielsen, *Defunctionalization at Work* (2001). The shallow-to-deep
  transformation of section 2.3, as a general technique.
- **Deforestation.** Philip Wadler, *Deforestation: Transforming Programs
  to Eliminate Trees* (1990). Section 3.1's optimization, done statically
  by a compiler rather than dynamically by generators.
- **The iterator (pull) model.** Goetz Graefe, *Volcano — An Extensible and
  Parallel Query Evaluation System* (1994). Where `lazy_pull.py`'s
  architecture comes from.
- **Compiling query plans, produce/consume.** Thomas Neumann, *Efficiently
  Compiling Efficient Query Plans for Modern Hardware* (VLDB 2011). The
  pattern `codegen.py` implements, targeting LLVM instead of Python.
- **Vectorized execution.** Peter Boncz, Marcin Zukowski, and Niels Nes,
  *MonetDB/X100: Hyper-Pipelining Query Execution* (CIDR 2005). The model
  behind `vectorized.py`, with batches instead of whole columns.
- **Compilation versus vectorization.** Timo Kersten et al., *Everything
  You Always Wanted to Know About Compiled and Vectorized Queries But Were
  Afraid to Ask* (VLDB 2018). Sections 3.4 and 3.5 as rivals, measured
  properly.
- **The expression problem.** Philip Wadler's 1998 note of that name (widely
  available online) states the trade in section 4 precisely.
- **Language-integrated query.** Erik Meijer, Brian Beckman, and Gavin
  Bierman, *LINQ: Reconciling Objects, Relations and XML in the .NET
  Framework* (SIGMOD 2006). A production-scale version of the
  fluent-frontend-over-plan design, with language support.
- **Multi-stage programming.** Walid Taha and Tim Sheard, *MetaML and
  Multi-Stage Programming with Explicit Annotations* (2000). What
  section 3.4's quasiquotation looks like when the language supports it
  natively.

Modern systems to read alongside: Polars (lazy frontend, plan, optimizer —
`fluent_pushdown.py` at scale), DuckDB (vectorized push-based execution),
and pyarrow's Acero engine (`arrow.py`'s backend).
