from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast


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
        columns: dict[str, list[Any]] = {}
        for row in compile_plan(optimize(self._plan)).run():
            for col, val in row.items():
                columns.setdefault(col, []).append(val)
        return DataFrame(columns)


def generated_source(query: Query) -> str:
    """The Python source collect() would compile and run for this query."""
    plan = IntermediateResult.lift(query)._plan
    return compile_plan(optimize(plan)).source


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


# --- Compilation (replaces the interpreter's execute) ---
#
# The optimized plan is compiled to Python source and exec'd, instead of
# being interpreted. Each pipeline becomes one generator function with the
# source scan, filters, join probes, and limit counter fused into a single
# nest of loops; a join's build side is a pipeline of its own, so it gets
# its own nested generator (and a `return` for limit always stops exactly
# one pipeline). Values the generated code needs at run time but that have
# no source-text form -- the filter lambdas and the source column data --
# are bound to fresh names in an environment dict that the compiler builds
# alongside the source and that becomes the globals of the exec'd code.


@dataclass(frozen=True)
class CompiledQuery:
    source: str
    env: dict[str, Any]
    entry: str

    def run(self) -> Iterator[dict[str, Any]]:
        namespace = dict(self.env)
        exec(compile(self.source, "<generated kernel>", "exec"), namespace)
        return cast(Iterator[dict[str, Any]], namespace[self.entry]())


class Emitter:
    """Accumulates generated source lines and the environment they refer to."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._counters: dict[str, int] = {}
        self.env: dict[str, Any] = {}

    def fresh(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return f"{prefix}_{n}"

    def bind(self, prefix: str, value: Any) -> str:
        """Name a runtime value so generated code can refer to it."""
        name = self.fresh(prefix)
        self.env[name] = value
        return name

    def line(self, text: str) -> None:
        self._lines.append("    " * self._indent + text)

    def blank(self) -> None:
        self._lines.append("")

    @contextmanager
    def block(self, header: str) -> Iterator[None]:
        self.line(header)
        self._indent += 1
        yield
        self._indent -= 1

    def source(self) -> str:
        return "\n".join(self._lines) + "\n"


def compile_plan(plan: Plan) -> CompiledQuery:
    emitter = Emitter()
    entry = emit_kernel(plan, emitter)
    return CompiledQuery(emitter.source(), emitter.env, entry)


def emit_kernel(plan: Plan, e: Emitter) -> str:
    """Emit one generator function fusing the whole pipeline rooted at plan."""
    name = e.fresh("kernel")
    with e.block(f"def {name}():"):
        emit_pipeline(plan, e, consume=lambda row: e.line(f"yield {row}"))
    e.blank()
    return name


def emit_pipeline(plan: Plan, e: Emitter, consume: Callable[[str], None]) -> None:
    """Emit code producing plan's rows, calling consume to emit each row's use.

    consume is given the variable name holding the current row; whatever it
    emits lands inside the loops and tests emitted here. Operators above
    this plan node thus become code inside its loop nest: a fused kernel.
    """
    match plan:
        case Source(columns):
            src = e.bind("src", columns)
            # The row count is known at compile time, so bake it in.
            n = len(next(iter(columns.values())))
            i = e.fresh("i")
            row = e.fresh("row")
            with e.block(f"for {i} in range({n}):"):
                e.line(f"{row} = {{col: vals[{i}] for col, vals in {src}.items()}}")
                consume(row)
        case Filter(child, column, predicate):
            pred = e.bind("pred", predicate)

            def consume_filtered(row: str) -> None:
                with e.block(f"if {pred}({row}[{column!r}]):"):
                    consume(row)

            emit_pipeline(child, e, consume_filtered)
        case Join(left, right, left_on, right_on):
            # The build side is a separate pipeline: emit it as its own
            # nested generator and materialize it before the probe loop.
            e.blank()
            right_kernel = emit_kernel(right, e)
            rows = e.fresh("rows")
            e.line(f"{rows} = list({right_kernel}())")

            def consume_matches(left_row: str) -> None:
                right_row = e.fresh("row")
                with (e.block(f"for {right_row} in {rows}:"),
                      e.block(f"if {left_row}[{left_on!r}] == {right_row}[{right_on!r}]:")):
                    joined = e.fresh("row")
                    e.line(f"{joined} = {left_row} | {right_row}")
                    consume(joined)

            emit_pipeline(left, e, consume_matches)
        case Limit(child, n):
            if n <= 0:
                # Constant-fold: nothing flows, so don't compile the child.
                e.line("return")
                e.line("yield  # unreachable; makes this function a generator")
                return
            count = e.fresh("count")
            e.line(f"{count} = 0")

            def consume_counted(row: str) -> None:
                consume(row)
                e.line(f"{count} += 1")
                with e.block(f"if {count} == {n}:"):
                    e.line("return")

            emit_pipeline(child, e, consume_counted)
