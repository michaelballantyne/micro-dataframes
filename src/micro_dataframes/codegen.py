import ast
import copy
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

    # The generated kernel source, for inspection.
    def generated_source(self) -> str:
        source, _ = compile_plan(optimize(self._plan))
        return source


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


# --- Quasiquotation over the stdlib ast module ---
#
# quote(template, **subs) parses a template string into Python AST and
# substitutes the keyword arguments into it: the template is the quoted
# code, the substitutions are the unquotes.  A placeholder is a bare Name:
#
#   - used as an expression, it is replaced by the given ast.expr;
#   - alone as an expression statement, it is replaced by the given
#     list of statements (splicing them into the enclosing block).
#
# Identifiers we mint ourselves (loop indices, counters, injected names)
# don't need placeholders: callers interpolate them into the template with
# an f-string.  Placeholders are for the two things f-strings can't carry
# safely: arbitrary data (rendered as ast.Constant) and previously built
# AST fragments.

_SubVal = ast.expr | list[ast.stmt]


class _QuoteTransformer(ast.NodeTransformer):
    def __init__(self, subs: dict[str, _SubVal]) -> None:
        self._subs = subs

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt]:
        # A statement consisting solely of a placeholder Name splices in a
        # block of statements.  (Returning a list from a visit method makes
        # NodeTransformer splice it into the parent's statement list.)
        if isinstance(node.value, ast.Name):
            val = self._subs.get(node.value.id)
            if isinstance(val, list):
                return val
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr:
        # A placeholder Name in expression position is replaced.  Deep-copy
        # so the same fragment can appear at several places in the tree.
        val = self._subs.get(node.id)
        if isinstance(val, ast.expr):
            return copy.deepcopy(val)
        return node


def quote(template: str, /, **subs: _SubVal) -> list[ast.stmt]:
    tree = _QuoteTransformer(subs).visit(ast.parse(template))
    assert isinstance(tree, ast.Module)
    return tree.body


def quote_expr(template: str, /, **subs: _SubVal) -> ast.expr:
    tree = _QuoteTransformer(subs).visit(ast.parse(template, mode="eval"))
    assert isinstance(tree, ast.Expression)
    return tree.body


# --- Code generation (produce/consume) ---
#
# produce(plan, consume, ...) builds the kernel's statements recursively.
# Each plan node contributes its outer structure (a loop header, an
# if-guard) and calls consume(col_vars) at the point where one row is
# available; consume is supplied by the parent operator and returns the
# statements the parent wants run for that row.  The outermost consume
# appends to the output columns.  Each pipeline ends up as one loop nest
# with no intermediate row representation: that is the fusion (the pattern
# comes from Neumann's HyPer compiler).  Joins break the pipeline: the
# right side becomes its own loop nest that fills buffers and an index dict
# keyed by the join column; the left pipeline then probes the index with a
# .get() lookup.
#
# col_vars maps each column name to the AST expression that reads its
# current value (e.g. source0['x'][_i0]).  No locals are bound in the
# kernel: filters, join probes, and output appends all use the expressions
# directly, so a row rejected by a filter costs only the predicate call.
#
# Runtime values that can't appear as literals in source code — the column
# dicts and the predicate functions — are injected into the namespace the
# kernel is exec'd in, under generated names (source0, pred0, ...).  The
# generated code refers to them by name.

# A consume callback takes the current col_vars and returns the inner body.
Consume = Callable[[dict[str, ast.expr]], list[ast.stmt]]


# An AST literal for {col: [], ...}.
def _empty_columns(cols: list[str]) -> ast.expr:
    return ast.Dict(
        keys=[ast.Constant(col) for col in cols],
        values=[ast.List(elts=[]) for _ in cols],
    )


# Statements appending each column's current value to target[col].
def _append_columns(target: str, cols: list[str], cv: dict[str, ast.expr]) -> list[ast.stmt]:
    stmts: list[ast.stmt] = []
    for col in cols:
        stmts += quote(f"{target}[COL].append(VAL)", COL=ast.Constant(col), VAL=cv[col])
    return stmts


class _NameSupply:
    # Fresh generated names: fresh("pred") gives pred0, pred1, ...
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def fresh(self, prefix: str) -> str:
        n = self._counts.get(prefix, 0)
        self._counts[prefix] = n + 1
        return f"{prefix}{n}"


def produce(
    plan: Plan,
    consume: Consume,
    ns: dict[str, Any],
    names: _NameSupply,
    col_vars: dict[str, ast.expr],
) -> list[ast.stmt]:
    match plan:
        case Source(columns):
            src = names.fresh("source")
            i = names.fresh("_i")
            # Inject the columns dict; the kernel indexes it as
            # source0['col'][_i0], so column names appear literally.
            ns[src] = columns
            first = next(iter(columns))

            new_cv = dict(col_vars)
            for col in columns:
                new_cv[col] = quote_expr(f"{src}[COL][{i}]", COL=ast.Constant(col))

            return quote(
                f"for {i} in range(len({src}[FIRST])):\n    BODY",
                FIRST=ast.Constant(first),
                BODY=consume(new_cv),
            )

        case Filter(child, column, predicate):
            pred = names.fresh("pred")
            ns[pred] = predicate

            def guarded(cv: dict[str, ast.expr]) -> list[ast.stmt]:
                return quote(
                    f"if {pred}(VAL):\n    BODY",
                    VAL=cv[column],
                    BODY=consume(cv),
                )

            return produce(child, guarded, ns, names, col_vars)

        case Join(left, right, left_on, right_on):
            # Build: index the right side by key.  Probe: stream the left side.
            # The right-side pipeline runs first, materializes into column buffers,
            # then a second build step indexes the buffer by the key column.  The
            # left pipeline probes the index with a .get() lookup.  Both pipelines
            # are generated code in the same kernel.
            buf = names.fresh("_right")
            fill = names.fresh("_fill")
            right_cols = sorted(schema(right))

            # The build pipeline gets its own function so that an early exit
            # (a limit's `return out` somewhere in the right side) stops only
            # the buffer filling, not the whole kernel.
            build = quote(f"{buf} = INIT", INIT=_empty_columns(right_cols)) + quote(
                f"def {fill}():\n    BODY\n{fill}()",
                BODY=produce(
                    right, lambda cv: _append_columns(buf, right_cols, cv), ns, names, {}
                ),
            )
            idx = names.fresh("_idx")
            j = names.fresh("_i")
            build += quote(
                f"{idx} = {{}}\n"
                f"for {j} in range(len({buf}[KEY])):\n"
                f"    {idx}.setdefault({buf}[KEY][{j}], []).append({j})",
                KEY=ast.Constant(right_on),
            )
            jp = names.fresh("_i")

            def join_consume(left_cv: dict[str, ast.expr]) -> list[ast.stmt]:
                right_cv = {
                    col: quote_expr(f"{buf}[COL][{jp}]", COL=ast.Constant(col))
                    for col in right_cols
                }
                # Columns are assumed disjoint, so the dict merge is collision-free.
                merged = left_cv | right_cv
                return quote(
                    f"for {jp} in {idx}.get(KEY, []):\n    BODY",
                    KEY=left_cv[left_on],
                    BODY=consume(merged),
                )

            return build + produce(left, join_consume, ns, names, col_vars)

        case Limit(child, n):
            ctr = names.fresh("_lim")

            def limited(cv: dict[str, ast.expr]) -> list[ast.stmt]:
                return quote(
                    f"if {ctr} == N:\n    return out\n{ctr} += 1\nBODY",
                    N=ast.Constant(n),
                    BODY=consume(cv),
                )

            return quote(f"{ctr} = 0") + produce(child, limited, ns, names, col_vars)


# The generated kernel source for a plan, for inspection.
def generated_source(plan: Plan) -> str:
    source, _ = compile_plan(plan)
    return source


# Compile an optimized plan into a fused kernel.  Returns (source, kernel):
# the unparsed source for inspection, and the kernel, which returns a
# columns dict ready to wrap in a DataFrame.
def compile_plan(plan: Plan) -> tuple[str, Callable[[], dict[str, list[Any]]]]:
    ns: dict[str, Any] = {}
    names = _NameSupply()
    out_cols = sorted(schema(plan))

    module = ast.Module(
        body=quote(
            "def kernel():\n    out = INIT\n    BODY\n    return out",
            INIT=_empty_columns(out_cols),
            # The innermost consume appends each column's value expression.
            BODY=produce(plan, lambda cv: _append_columns("out", out_cols, cv), ns, names, {}),
        ),
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    source = ast.unparse(module)
    exec(compile(module, "<plan-codegen>", "exec"), ns)
    kernel: Callable[[], dict[str, list[Any]]] = ns["kernel"]
    return source, kernel
