from __future__ import annotations

from collections import defaultdict


def evaluate_dnf(formula,true_props):
    """
    Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false. 
    e.g. evaluate_dnf("a&b|!c&d","d") returns True 
    """
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if evaluate_dnf(f,true_props):
                return True
        return False
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not evaluate_dnf(f,true_props):
                return False
        return True
    # NOT
    if formula.startswith("!"):
        return not evaluate_dnf(formula[1:],true_props)

    if formula.startswith("("):
        return evaluate_dnf(formula[1:], true_props)

    if formula.endswith(")"):
        return evaluate_dnf(formula[:-1], true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props


def _tokenize_formula(formula: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(formula):
        c = formula[i]
        if c.isspace():
            i += 1
            continue
        if c in {"(", ")", "!", "&", "|"}:
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < len(formula) and (formula[j].isalnum() or formula[j] == "_"):
            j += 1
        if j == i:
            raise ValueError(f"Unexpected character '{formula[i]}' in formula '{formula}'")
        tokens.append(formula[i:j])
        i = j
    return tokens


class _FormulaParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self):
        node = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError("Unexpected trailing tokens in formula")
        return node

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, token: str) -> None:
        if self._peek() != token:
            raise ValueError(f"Expected '{token}' but got '{self._peek()}'")
        self.pos += 1

    def _parse_or(self):
        node = self._parse_and()
        while self._peek() == "|":
            self.pos += 1
            node = ("or", node, self._parse_and())
        return node

    def _parse_and(self):
        node = self._parse_not()
        while self._peek() == "&":
            self.pos += 1
            node = ("and", node, self._parse_not())
        return node

    def _parse_not(self):
        if self._peek() == "!":
            self.pos += 1
            return ("not", self._parse_not())
        return self._parse_atom()

    def _parse_atom(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        if tok == "(":
            self.pos += 1
            node = self._parse_or()
            self._consume(")")
            return node
        self.pos += 1
        if tok == "True":
            return ("const", True)
        if tok == "False":
            return ("const", False)
        return ("var", tok)


def _collect_vars(node, out: set[str]) -> None:
    kind = node[0]
    if kind == "var":
        out.add(node[1])
    elif kind in {"and", "or"}:
        _collect_vars(node[1], out)
        _collect_vars(node[2], out)
    elif kind == "not":
        _collect_vars(node[1], out)


def _eval_ast(node, assignment: dict[str, bool]) -> bool:
    kind = node[0]
    if kind == "const":
        return bool(node[1])
    if kind == "var":
        return assignment.get(node[1], False)
    if kind == "not":
        return not _eval_ast(node[1], assignment)
    if kind == "and":
        return _eval_ast(node[1], assignment) and _eval_ast(node[2], assignment)
    if kind == "or":
        return _eval_ast(node[1], assignment) or _eval_ast(node[2], assignment)
    raise ValueError(f"Unsupported AST node: {kind}")


def _qm_combine(a: str, b: str) -> str | None:
    diff = 0
    out: list[str] = []
    for ca, cb in zip(a, b):
        if ca == cb:
            out.append(ca)
            continue
        if ca == "-" or cb == "-":
            return None
        diff += 1
        if diff > 1:
            return None
        out.append("-")
    if diff != 1:
        return None
    return "".join(out)


def _term_covers(pattern: str, minterm: int, n: int) -> bool:
    for i, ch in enumerate(pattern):
        if ch == "-":
            continue
        bit = (minterm >> (n - 1 - i)) & 1
        if ch == "1" and bit != 1:
            return False
        if ch == "0" and bit != 0:
            return False
    return True


def _literal_count(pattern: str) -> int:
    return sum(1 for c in pattern if c != "-")


def _pattern_to_term(pattern: str, variables: list[str]) -> str:
    literals: list[str] = []
    for bit, var in zip(pattern, variables):
        if bit == "1":
            literals.append(var)
        elif bit == "0":
            literals.append(f"!{var}")
    if not literals:
        return "True"
    return "&".join(literals)


def simplify_formula_to_dnf(formula: str, simplify: bool = True) -> str:
    """
    Convert a boolean formula to a simplified DNF formula.
    """
    tokens = _tokenize_formula(formula)
    ast = _FormulaParser(tokens).parse()

    vars_set: set[str] = set()
    _collect_vars(ast, vars_set)
    variables = sorted(vars_set)
    n = len(variables)

    if n == 0:
        return "True" if _eval_ast(ast, {}) else "False"

    minterms: list[int] = []
    for mask in range(1 << n):
        assignment = {variables[i]: bool((mask >> (n - 1 - i)) & 1) for i in range(n)}
        if _eval_ast(ast, assignment):
            minterms.append(mask)

    if not minterms:
        return "False"
    if len(minterms) == (1 << n):
        return "True"

    current = {format(m, f"0{n}b"): {m} for m in minterms}
    prime_implicants: dict[str, set[int]] = {}

    while current:
        grouped: dict[int, list[str]] = defaultdict(list)
        for p in current:
            grouped[p.count("1")].append(p)

        used: set[str] = set()
        next_map: dict[str, set[int]] = {}
        keys = sorted(grouped.keys())
        for k in keys:
            if k + 1 not in grouped:
                continue
            for a in grouped[k]:
                for b in grouped[k + 1]:
                    c = _qm_combine(a, b)
                    if c is None:
                        continue
                    used.add(a)
                    used.add(b)
                    next_map.setdefault(c, set()).update(current[a] | current[b])

        for p, covered in current.items():
            if p not in used:
                prime_implicants[p] = covered

        current = next_map

    coverage: dict[int, list[str]] = {m: [] for m in minterms}
    for p, covered in prime_implicants.items():
        for m in covered:
            if m in coverage:
                coverage[m].append(p)

    selected: set[str] = set()
    remaining = set(minterms)

    for m, implicants in coverage.items():
        if len(implicants) == 1:
            selected.add(implicants[0])

    for p in selected:
        remaining = {m for m in remaining if m not in prime_implicants[p]}

    candidates = [p for p in prime_implicants if p not in selected and any(m in remaining for m in prime_implicants[p])]
    best: tuple[tuple[int, int, tuple[str, ...]], set[str]] | None = None

    if remaining:
    # Exact search over subsets of remaining prime implicants.
        # Use a light guard to avoid combinatorial blow-up.
        if len(candidates) <= 16:
            from itertools import combinations

            for r in range(1, len(candidates) + 1):
                for combo in combinations(candidates, r):
                    covered = set()
                    for p in combo:
                        covered.update(prime_implicants[p] & remaining)
                    if covered != remaining:
                        continue
                    score = (
                        len(combo),
                        sum(_literal_count(p) for p in combo),
                        tuple(sorted(combo)),
                    )
                    if best is None or score < best[0]:
                        best = (score, set(combo))
                if best is not None:
                    break
        else:
            # Greedy fallback for large implicant sets.
            chosen: set[str] = set()
            rem = set(remaining)
            while rem:
                pick = max(
                    candidates,
                    key=lambda p: (len(prime_implicants[p] & rem), -_literal_count(p)),
                )
                chosen.add(pick)
                rem -= prime_implicants[pick]
                candidates = [p for p in candidates if p != pick]
            best = ((len(chosen), sum(_literal_count(p) for p in chosen), tuple(sorted(chosen))), chosen)

    if best is not None:
        selected |= best[1]

    if simplify:
        terms = sorted(selected, key=lambda p: (_literal_count(p), p))
        if not terms:
            return "False"

        dnf_terms = [_pattern_to_term(p, variables) for p in terms]
        if "True" in dnf_terms:
            return "True"
        if len(dnf_terms) == 1:
            return dnf_terms[0]
        return "|".join(f"({t})" for t in dnf_terms)

    # Canonical (not minimized) DNF: one full minterm per satisfying assignment.
    canonical_terms = []
    for mask in sorted(minterms):
        bits = format(mask, f"0{n}b")
        canonical_terms.append(_pattern_to_term(bits, variables))
    if len(canonical_terms) == 1:
        return canonical_terms[0]
    return "|".join(f"({t})" for t in canonical_terms)

def value_iteration(states, delta_u, delta_r, terminal_u, gamma):
    """
    Standard value iteration approach. 
    We use it to compute the potential function for the automated reward shaping
    """
    v_values = dict([(u,0) for u in states])
    v_values[terminal_u] = 0
    v_errors = 1
    while v_errors > 0.0000001:
        v_errors = 0
        for u1 in states:
            q_u2 = []
            for u2 in delta_u[u1]:
                if delta_r[u1][u2].get_type() == "constant": 
                    r = delta_r[u1][u2].get_reward(None)
                else:
                    r = 0 # If the reward function is not constant, we assume it returns a reward of zero
                q_u2.append(r+gamma*v_values[u2])
            v_new = max(q_u2)
            v_errors = max([v_errors, abs(v_new-v_values[u1])])
            v_values[u1] = v_new
    return v_values
