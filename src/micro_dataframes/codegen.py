from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class Query(Protocol):
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult: ...
    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult: ...
    def limit(self, n: int) -> IntermediateResult: ...
    def collect(self) -> DataFrame: ...


class DataFrame(Query):
    _columns: dict[str, list[Any]]

    def __init__(self, data: dict[str, list[Any]] | list[dict[str, Any]]) -> None:
        if isinstance(data, list):
            self._columns = {}
            for row in data:
                for key, value in row.items():
                    if key not in self._columns:
                        self._columns[key] = []
                    self._columns[key].append(value)
        else:
            self._columns = data

    # Forward Query methods to IntermediateResult.
    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        return IntermediateResult.lift(self).filter(column, predicate)

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        return IntermediateResult.lift(self).join(other, left_on, right_on)

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult.lift(self).limit(n)

    # Except collect.
    def collect(self) -> DataFrame:
        return self

    def __getitem__(self, column: str) -> list[Any]:
        return self._columns[column]


class IntermediateResult(Query):
    _plan: Plan

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    @classmethod
    def lift(cls, query: Query) -> IntermediateResult:
        # Keeping an IntermediateResult's plan intact (rather than
        # collecting it) is what lets the optimizer push filters into
        # the right side of a join.
        if isinstance(query, IntermediateResult):
            return query
        return cls(Source(query.collect()._columns))

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> IntermediateResult:
        return IntermediateResult(Filter(self._plan, column, predicate))

    def join(self, other: Query, left_on: str, right_on: str) -> IntermediateResult:
        other_plan = IntermediateResult.lift(other)._plan
        return IntermediateResult(Join(self._plan, other_plan, left_on, right_on))

    def limit(self, n: int) -> IntermediateResult:
        return IntermediateResult(Limit(self._plan, n))

    def collect(self) -> DataFrame:
        _, kernel = compile_plan(optimize(self._plan))
        return DataFrame(kernel())

    def generated_source(self) -> str:
        """Return the generated kernel source so it can be printed and inspected."""
        src, _ = compile_plan(optimize(self._plan))
        return src


# --- Plan nodes (the deep embedding) ---

type Plan = Source | Filter | Join | Limit


@dataclass(frozen=True)
class Source:
    columns: dict[str, list[Any]]

    def __repr__(self) -> str:
        return f"Source(columns={sorted(self.columns.keys())})"


@dataclass(frozen=True)
class Filter:
    child: Plan
    column: str
    predicate: Callable[[Any], bool]


@dataclass(frozen=True)
class Join:
    left: Plan
    right: Plan
    left_on: str
    right_on: str


@dataclass(frozen=True)
class Limit:
    child: Plan
    n: int


# --- Passes over plans ---

def schema(plan: Plan) -> set[str]:
    match plan:
        case Source(columns):
            return set(columns.keys())
        case Filter(child, _, _):
            return schema(child)
        case Join(left, right, _, _):
            return schema(left) | schema(right)
        case Limit(child, _):
            return schema(child)


def optimize(plan: Plan) -> Plan:
    match plan:
        case Filter(child, column, predicate):
            # Push a filter below a join when its column comes from one side.
            match optimize(child):
                case Join(left, right, left_on, right_on) if column in schema(left):
                    return Join(optimize(Filter(left, column, predicate)),
                                right, left_on, right_on)
                case Join(left, right, left_on, right_on) if column in schema(right):
                    return Join(left,
                                optimize(Filter(right, column, predicate)),
                                left_on, right_on)
                case child:
                    return Filter(child, column, predicate)
        case Join(left, right, left_on, right_on):
            return Join(optimize(left), optimize(right), left_on, right_on)
        case Limit(child, n):
            return Limit(optimize(child), n)
        case Source():
            return plan


# --- Code generation (Neumann-style produce/consume) ---
#
# Pattern: produce(plan, consume, ...) recursively emits code for a plan node.
# Each node emits its *outer* structure (loop headers, if-guards) and then
# calls consume(col_vars) at the point where one logical row is available.
# consume is a callback supplied by the *parent* operator that emits whatever
# code the parent needs to do with that row.  The outermost consume emits the
# per-column out[col].append(...) calls.  The whole pipeline collapses into a
# single flat function with no intermediate row dicts — that is the fusion.
#
# col_vars: dict[column_name -> Python expression string] carries the current
# binding of each column as a *source expression* (e.g. "source0['x'][_i0]").
# No intermediate locals are emitted: filters guard on the expression directly,
# joins compare expressions, and the output appends expressions.  This removes
# all dead binds — a row rejected by a filter costs only the predicate call —
# and makes the generated code self-describing since column names appear
# literally in the source.
#
# Join fusion: produce(left, ...) receives a consume callback that recursively
# calls produce(right, inner, ...).  inner emits the equality guard and then
# calls the parent consume with merged col_vars (right side wins name
# collisions, mirroring `left_row | right_row`).  The generated kernel contains
# all the nested loops directly; pushed-down right-side filters appear as
# if-guards inside the inner loop.
#
# Trade-off: the right side of a join is re-scanned for every left row (a real
# engine would break the pipeline and materialise the build side into a hash
# table; here we keep everything in one fused kernel for simplicity).
#
# Runtime values that cannot be rendered as literals (column dicts, predicates)
# are injected into the kernel's exec namespace via `ns`.  Generated code
# references them by their injected names.


# A consume callback receives the current col_vars mapping and emits the inner body.
Consume = Callable[[dict[str, str]], None]


class _Emitter:
    """Accumulates source lines with indentation tracking."""

    _lines: list[str]
    _depth: int

    def __init__(self) -> None:
        self._lines = []
        self._depth = 0

    def emit(self, line: str) -> None:
        self._lines.append("    " * self._depth + line)

    def indent(self) -> None:
        self._depth += 1

    def dedent(self) -> None:
        self._depth -= 1

    def source(self) -> str:
        return "\n".join(self._lines)


@dataclass
class _State:
    """Monotonically-increasing counters for fresh generated names."""
    src: int = 0    # source0, source1, …  (injected column dicts)
    pred: int = 0   # pred0, pred1, …  (injected predicate callables)
    loop: int = 0   # _i0, _i1, …  (loop index variables, one per Source/Join-right)
    lim: int = 0    # _lim0, _lim1, …  (limit counter variables)

    def fresh_src(self) -> str:
        name = f"source{self.src}"
        self.src += 1
        return name

    def fresh_pred(self) -> str:
        name = f"pred{self.pred}"
        self.pred += 1
        return name

    def fresh_loop(self) -> str:
        name = f"_i{self.loop}"
        self.loop += 1
        return name

    def fresh_lim(self) -> str:
        name = f"_lim{self.lim}"
        self.lim += 1
        return name


def produce(
    plan: Plan,
    consume: Consume,
    emit: _Emitter,
    ns: dict[str, Any],
    st: _State,
    col_vars: dict[str, str],
) -> None:
    """Emit code for *plan*, calling consume(col_vars) for each output row."""
    match plan:
        case Source(columns):
            src = st.fresh_src()
            idx = st.fresh_loop()
            # Inject the columns dict directly; generated code indexes it as
            # source0['col'][_i0] so column names appear literally in the kernel.
            ns[src] = columns
            # Use the first column's list to determine the row count.
            first_col = next(iter(columns))

            emit.emit(f"for {idx} in range(len({src}[{first_col!r}])):")
            emit.indent()

            new_cv = dict(col_vars)
            for col in columns:
                new_cv[col] = f"{src}[{col!r}][{idx}]"

            consume(new_cv)
            emit.dedent()

        case Filter(child, column, predicate):
            pred = st.fresh_pred()
            ns[pred] = predicate

            # Wrap the downstream consume in an if-guard; col_vars is a
            # plain closure variable (not a loop variable) so default-arg
            # capture is unnecessary.
            def guarded(cv: dict[str, str]) -> None:
                emit.emit(f"if {pred}({cv[column]}):")
                emit.indent()
                consume(cv)
                emit.dedent()

            produce(child, guarded, emit, ns, st, col_vars)

        case Join(left, right, left_on, right_on):
            # Fused nested-loop join: produce the left side, and inside its
            # consume produce the right side with an equality guard.
            # Trade-off noted above: right side is rescanned per left row.
            def join_consume(left_cv: dict[str, str]) -> None:
                left_expr = left_cv[left_on]

                def inner(right_cv: dict[str, str]) -> None:
                    right_expr = right_cv[right_on]
                    emit.emit(f"if {left_expr} == {right_expr}:")
                    emit.indent()
                    # Right wins on name collision (mirrors left_row | right_row).
                    merged = left_cv | right_cv
                    consume(merged)
                    emit.dedent()

                produce(right, inner, emit, ns, st, left_cv)

            produce(left, join_consume, emit, ns, st, col_vars)

        case Limit(child, n):
            ctr = st.fresh_lim()
            emit.emit(f"{ctr} = 0")

            def limited(cv: dict[str, str]) -> None:
                emit.emit(f"if {ctr} == {n}:")
                emit.indent()
                emit.emit("return out")
                emit.dedent()
                emit.emit(f"{ctr} += 1")
                consume(cv)

            produce(child, limited, emit, ns, st, col_vars)


def generated_source(plan: Plan) -> str:
    """Return the generated kernel source for a plan without executing it."""
    src, _ = compile_plan(plan)
    return src


def compile_plan(plan: Plan) -> tuple[str, Callable[[], dict[str, list[Any]]]]:
    """Compile an optimised plan into a fused Python kernel.

    Returns *(source, kernel)*.  ``source`` is the generated source for
    inspection; ``kernel()`` executes it and returns a column-oriented dict
    suitable for passing directly to DataFrame(...).
    """
    emit = _Emitter()
    ns: dict[str, Any] = {}
    st = _State()

    out_cols = sorted(schema(plan))

    emit.emit("def kernel():")
    emit.indent()
    # Initialise one output list per schema column.
    out_init = "{" + ", ".join(f"{col!r}: []" for col in out_cols) + "}"
    emit.emit(f"out = {out_init}")

    # The innermost consume: append each column's expression to out[col].
    def emit_row(cv: dict[str, str]) -> None:
        for col in out_cols:
            if col in cv:
                emit.emit(f"out[{col!r}].append({cv[col]})")

    produce(plan, emit_row, emit, ns, st, col_vars={})

    emit.emit("return out")
    emit.dedent()

    source = emit.source()
    code = compile(source, "<plan-codegen>", "exec")
    exec(code, ns)
    kernel: Callable[[], dict[str, list[Any]]] = ns["kernel"]
    return source, kernel
