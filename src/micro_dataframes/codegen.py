# Plan AST -> Python AST -> compile
#
# The pipeline is:
#   Plan (dataclass AST) -> produce() -> Python ast.Module -> compile() -> kernel()
#
# Runtime values that cannot be represented as literals (column dicts,
# predicates) are injected into the kernel's exec namespace under generated
# names (source0, pred0, …) exactly as before.  The kernel is compiled with
# compile(module, "<plan-codegen>", "exec") and inspectable via ast.unparse().

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


# --- Quasiquote helpers ---
#
# quote/quote_expr implement quasiquotation in ~30 lines of stdlib:
# the template string is the "quote" (a literal AST fragment) and the
# keyword arguments are the "unquotes" (substitutions spliced into it).
#
# quote(template, **subs) -> list[ast.stmt]
#   Parse the template with ast.parse, walk it with a NodeTransformer:
#   - A Name whose id is a key in subs is replaced by the given ast.expr.
#   - An Expr statement whose sole child is such a Name is *spliced* with
#     the given list[ast.stmt], flattening it into the enclosing block.
#   Substituted nodes are deep-copied so the same expression can safely
#   appear in multiple places in the tree.
#
# quote_expr(template, **subs) -> ast.expr
#   Same idea for a single expression (ast.parse mode="eval").

# Substitution value: either a single expression or a block of statements.
_SubVal = ast.expr | list[ast.stmt]
_SubMap = dict[str, _SubVal]


class _QuoteTransformer(ast.NodeTransformer):
    """Walk a parsed template and splice in unquoted sub-expressions/statements."""

    def __init__(self, subs: _SubMap) -> None:
        self._subs = subs

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt]:
        # If this expression-statement is solely a placeholder Name, splice
        # the list[ast.stmt] substitution into the enclosing block.
        if isinstance(node.value, ast.Name):
            key = node.value.id
            if key in self._subs:
                val = self._subs[key]
                if isinstance(val, list):
                    return val  # splice: NodeTransformer extends the parent list
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr | ast.Name:
        # Replace a placeholder Name with an expression substitution.
        # When the substitution is itself a Name, inherit the template's ctx
        # so that Store/Del/Load positions remain syntactically correct (e.g.
        # the target of `CTR += 1` must keep Store, even though we pass Load).
        key = node.id
        if key in self._subs:
            val = self._subs[key]
            if isinstance(val, ast.expr):
                replacement = copy.deepcopy(val)
                if isinstance(replacement, ast.Name):
                    replacement.ctx = node.ctx
                return replacement
        return node


def quote(template: str, /, **subs: _SubVal) -> list[ast.stmt]:
    """Parse *template* and substitute placeholders with *subs*.

    Placeholders are bare ``Name`` nodes whose ``id`` matches a keyword
    argument key.  When the value is an ``ast.expr`` the Name is replaced
    in-place; when it is a ``list[ast.stmt]`` and the Name appears as the
    sole child of an ``Expr`` statement, the whole statement is *spliced*
    (replaced by the list of statements).  This is quasiquotation: the
    template is the quote, the keyword arguments are the unquotes.
    """
    tree = ast.parse(template)
    new_tree = _QuoteTransformer(subs).visit(tree)
    assert isinstance(new_tree, ast.Module)
    return new_tree.body


def quote_expr(template: str, /, **subs: _SubVal) -> ast.expr:
    """Parse *template* as a single expression and substitute placeholders."""
    tree = ast.parse(template, mode="eval")
    new_tree = _QuoteTransformer(subs).visit(tree)
    assert isinstance(new_tree, ast.Expression)
    return new_tree.body


# --- Code generation (Neumann-style produce/consume) ---
#
# Pattern: produce(plan, consume, ...) recursively builds AST statements for a
# plan node.  Each node emits its *outer* structure (loop headers, if-guards)
# and then calls consume(col_vars) at the point where one logical row is
# available.  consume is a callback supplied by the *parent* operator that
# returns whatever statements the parent needs to do with that row.  The
# outermost consume emits the per-column out[col].append(...) calls.  The
# whole pipeline collapses into a single flat function with no intermediate row
# dicts — that is the fusion.
#
# col_vars: dict[column_name -> ast.expr] carries the current binding of each
# column as a *source expression* (e.g. source0['x'][_i0] as an AST node).
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


# A consume callback receives the current col_vars mapping and returns the inner body.
Consume = Callable[[dict[str, ast.expr]], list[ast.stmt]]


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
    ns: dict[str, Any],
    st: _State,
    col_vars: dict[str, ast.expr],
) -> list[ast.stmt]:
    """Build AST statements for *plan*, calling consume(col_vars) for each output row."""
    match plan:
        case Source(columns):
            src = st.fresh_src()
            idx = st.fresh_loop()
            # Inject the columns dict directly; generated code indexes it as
            # source0['col'][_i0] so column names appear literally in the kernel.
            ns[src] = columns
            # Use the first column's list to determine the row count.
            first_col = next(iter(columns))

            new_cv = dict(col_vars)
            for col in columns:
                new_cv[col] = quote_expr(
                    "SRC[COL][I]",
                    SRC=ast.Name(src, ctx=ast.Load()),
                    COL=ast.Constant(col),
                    I=ast.Name(idx, ctx=ast.Load()),
                )

            body = consume(new_cv)
            return quote(
                "for I in range(len(SRC[FIRST])):\n    BODY",
                I=ast.Name(idx, ctx=ast.Load()),
                SRC=ast.Name(src, ctx=ast.Load()),
                FIRST=ast.Constant(first_col),
                BODY=body,
            )

        case Filter(child, column, predicate):
            pred = st.fresh_pred()
            ns[pred] = predicate

            # Wrap the downstream consume in an if-guard; col_vars is a
            # plain closure variable (not a loop variable) so default-arg
            # capture is unnecessary.
            def guarded(cv: dict[str, ast.expr]) -> list[ast.stmt]:
                body = consume(cv)
                return quote(
                    "if PRED(VAL):\n    BODY",
                    PRED=ast.Name(pred, ctx=ast.Load()),
                    VAL=cv[column],
                    BODY=body,
                )

            return produce(child, guarded, ns, st, col_vars)

        case Join(left, right, left_on, right_on):
            # Fused nested-loop join: produce the left side, and inside its
            # consume produce the right side with an equality guard.
            # Trade-off noted above: right side is rescanned per left row.
            def join_consume(left_cv: dict[str, ast.expr]) -> list[ast.stmt]:
                left_expr = left_cv[left_on]

                def inner(right_cv: dict[str, ast.expr]) -> list[ast.stmt]:
                    right_expr = right_cv[right_on]
                    # Right wins on name collision (mirrors left_row | right_row).
                    merged = left_cv | right_cv
                    body = consume(merged)
                    return quote(
                        "if L == R:\n    BODY",
                        L=left_expr,
                        R=right_expr,
                        BODY=body,
                    )

                return produce(right, inner, ns, st, left_cv)

            return produce(left, join_consume, ns, st, col_vars)

        case Limit(child, n):
            ctr = st.fresh_lim()

            def limited(cv: dict[str, ast.expr]) -> list[ast.stmt]:
                body = consume(cv)
                return quote(
                    "if CTR == N:\n    return out\nCTR += 1\nBODY",
                    CTR=ast.Name(ctr, ctx=ast.Load()),
                    N=ast.Constant(n),
                    BODY=body,
                )

            ctr_init = quote("CTR = 0", CTR=ast.Name(ctr, ctx=ast.Load()))
            return ctr_init + produce(child, limited, ns, st, col_vars)


def generated_source(plan: Plan) -> str:
    """Return the generated kernel source for a plan without executing it."""
    src, _ = compile_plan(plan)
    return src


def compile_plan(plan: Plan) -> tuple[str, Callable[[], dict[str, list[Any]]]]:
    """Compile an optimised plan into a fused Python kernel.

    Returns *(source, kernel)*.  ``source`` is the generated source for
    inspection (via ast.unparse); ``kernel()`` executes it and returns a
    column-oriented dict suitable for passing directly to DataFrame(...).
    """
    ns: dict[str, Any] = {}
    st = _State()

    out_cols = sorted(schema(plan))

    # Build the out = {col: [], ...} initialiser directly as an AST node.
    out_init = ast.Assign(
        targets=[ast.Name("out", ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant(col) for col in out_cols],
            values=[ast.List(elts=[], ctx=ast.Load()) for _ in out_cols],
        ),
        lineno=0,
        col_offset=0,
    )

    # The innermost consume: append each column's expression to out[col].
    def emit_row(cv: dict[str, ast.expr]) -> list[ast.stmt]:
        stmts: list[ast.stmt] = []
        for col in out_cols:
            if col in cv:
                stmts += quote(
                    "out[COL].append(VAL)",
                    COL=ast.Constant(col),
                    VAL=cv[col],
                )
        return stmts

    body_stmts = produce(plan, emit_row, ns, st, col_vars={})
    return_stmt = ast.Return(value=ast.Name("out", ctx=ast.Load()))

    func_def = ast.FunctionDef(
        name="kernel",
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        ),
        body=[out_init, *body_stmts, return_stmt],
        decorator_list=[],
        returns=None,
        lineno=0,
        col_offset=0,
    )

    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    source = ast.unparse(module)
    code = compile(module, "<plan-codegen>", "exec")
    exec(code, ns)
    kernel: Callable[[], dict[str, list[Any]]] = ns["kernel"]
    return source, kernel
