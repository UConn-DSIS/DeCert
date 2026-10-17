"""Pure scoring math for the decision-certificate reward.

The structured certificate induces a criterion-local depth-2 claim graph:
observation claims are leaves and each criterion conclusion is their child.
The functions below implement the paper's support-chain soundness,
reliability-aware maximum-weight coverage, conservative decision margin, and
overall score aggregation.
"""

from __future__ import annotations

from .types import Judgment, MaterialIssue

# Score maps for verifier verdicts.
GROUNDING_VALUE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
VALIDITY_VALUE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
MATCH_VALUE = {"full": 1.0, "partial": 0.5, "none": 0.0}


def claim_reliabilities(
    judgment: Judgment,
    grounding: dict[int, float],
    validity: dict[int, float],
) -> dict[int, float]:
    """Return support-chain validity ``q_j`` for every numbered claim.

    Claim numbering is the same as ``prompts.numbered_claims``. An observation
    is a leaf, so its support-chain validity is its grounding score. A
    criterion conclusion has all observations in its block as parents, so its
    validity is the minimum of its local validity and all parent validities.
    Missing verifier records take the pessimistic value zero.
    """
    out: dict[int, float] = {}
    claim_idx = 0
    observation_idx = 0
    for block_idx, block in enumerate(judgment.criteria, start=1):
        parent_q: list[float] = []
        for _obs in block.observations:
            claim_idx += 1
            observation_idx += 1
            q = grounding.get(observation_idx, 0.0)
            q = max(0.0, min(1.0, q))
            out[claim_idx] = q
            parent_q.append(q)

        claim_idx += 1
        local_q = max(0.0, min(1.0, validity.get(block_idx, 0.0)))
        out[claim_idx] = min([local_q, *parent_q]) if parent_q else local_q
    return out


def _criterion_claim_indices(judgment: Judgment) -> list[list[int]]:
    """Numbered claim indices grouped by criterion block."""
    grouped: list[list[int]] = []
    claim_idx = 0
    for block in judgment.criteria:
        indices = []
        for _obs in block.observations:
            claim_idx += 1
            indices.append(claim_idx)
        claim_idx += 1
        indices.append(claim_idx)  # criterion conclusion
        grouped.append(indices)
    return grouped


def criterion_qualities(
    judgment: Judgment,
    grounding: dict[int, float],
    validity: dict[int, float],
) -> list[float]:
    """Return support-chain validity of each criterion conclusion.

    This compatibility helper exposes the criterion-level ``q`` values used
    to construct verified intervals. The scoring path uses
    ``claim_reliabilities`` directly so all dimensions share the same values.
    """
    claim_q = claim_reliabilities(judgment, grounding, validity)
    return [claim_q[indices[-1]] for indices in _criterion_claim_indices(judgment)]


def soundness(
    judgment: Judgment,
    claim_q: dict[int, float],
    weights: dict[str, float],
) -> float:
    """Policy-weighted geometric mean of criterion bottleneck validity.

    For each policy criterion, ``s_k`` is the minimum support-chain validity
    among its claims. A criterion absent from the certificate contributes one,
    following the paper; the release parser normally rejects such omissions.
    """
    block_scores: dict[str, float] = {}
    for block, indices in zip(judgment.criteria, _criterion_claim_indices(judgment)):
        block_scores[block.name] = min(claim_q.get(i, 0.0) for i in indices)

    score = 1.0
    for name, weight in weights.items():
        score *= block_scores.get(name, 1.0) ** weight
    return score


def maximum_weight_matching(
    issues: list[MaterialIssue],
    pair_scores: list[list[float]],
    claim_q: dict[int, float],
) -> list[tuple[int, int | None, float]]:
    """Solve the maximum-weight partial one-to-one issue--claim matching.

    Edge ``(t, j)`` has weight ``rho_t * m_tj * q_j``. The analyzer emits at
    most six issues. We add one zero-weight dummy claim per issue and solve the
    resulting rectangular assignment with a pure-Python Hungarian algorithm.
    """
    n_claims = max(claim_q, default=0)
    if not issues:
        return []

    weights: list[list[float]] = []
    for issue_idx, issue in enumerate(issues):
        row = pair_scores[issue_idx] if issue_idx < len(pair_scores) else []
        real_edges = [
            issue.importance
            * (row[j] if j < len(row) else 0.0)
            * claim_q.get(j + 1, 0.0)
            for j in range(n_claims)
        ]
        weights.append([*real_edges, *([0.0] * len(issues))])

    assignment = _hungarian_minimize([[-value for value in row] for row in weights])
    matches = []
    for issue_zero_idx, claim_zero_idx in enumerate(assignment):
        row = pair_scores[issue_zero_idx] if issue_zero_idx < len(pair_scores) else []
        if claim_zero_idx < n_claims and weights[issue_zero_idx][claim_zero_idx] > 0.0:
            m = row[claim_zero_idx] if claim_zero_idx < len(row) else 0.0
            matches.append((issue_zero_idx + 1, claim_zero_idx + 1, m))
        else:
            matches.append((issue_zero_idx + 1, None, 0.0))
    return matches


def _hungarian_minimize(cost: list[list[float]]) -> list[int]:
    """Minimum-cost rectangular assignment for rows <= columns."""
    n_rows = len(cost)
    if n_rows == 0:
        return []
    n_cols = len(cost[0])
    if n_rows > n_cols or any(len(row) != n_cols for row in cost):
        raise ValueError("assignment matrix must be rectangular with rows <= columns")

    row_potential = [0.0] * (n_rows + 1)
    col_potential = [0.0] * (n_cols + 1)
    matched_row = [0] * (n_cols + 1)
    previous_col = [0] * (n_cols + 1)

    for row_idx in range(1, n_rows + 1):
        matched_row[0] = row_idx
        current_col = 0
        min_reduced = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[current_col] = True
            current_row = matched_row[current_col]
            delta = float("inf")
            next_col = 0
            for col_idx in range(1, n_cols + 1):
                if used[col_idx]:
                    continue
                reduced = (
                    cost[current_row - 1][col_idx - 1]
                    - row_potential[current_row]
                    - col_potential[col_idx]
                )
                if reduced < min_reduced[col_idx]:
                    min_reduced[col_idx] = reduced
                    previous_col[col_idx] = current_col
                if min_reduced[col_idx] < delta:
                    delta = min_reduced[col_idx]
                    next_col = col_idx

            for col_idx in range(n_cols + 1):
                if used[col_idx]:
                    row_potential[matched_row[col_idx]] += delta
                    col_potential[col_idx] -= delta
                elif col_idx > 0:
                    min_reduced[col_idx] -= delta
            current_col = next_col
            if matched_row[current_col] == 0:
                break

        while True:
            next_col = previous_col[current_col]
            matched_row[current_col] = matched_row[next_col]
            current_col = next_col
            if current_col == 0:
                break

    assignment = [-1] * n_rows
    for col_idx in range(1, n_cols + 1):
        if matched_row[col_idx] != 0:
            assignment[matched_row[col_idx] - 1] = col_idx - 1
    return assignment


def coverage(
    issues: list[MaterialIssue],
    matches: list[tuple[int, int | None, float]],
    claim_q: dict[int, float],
) -> float:
    """Reliability-aware importance-weighted material-issue coverage."""
    if not issues:
        raise ValueError("coverage is undefined for an empty material-issue set")
    total = sum(issue.importance for issue in issues)
    covered = 0.0
    for issue_idx, claim_idx, m in matches:
        if claim_idx is None:
            continue
        covered += issues[issue_idx - 1].importance * m * claim_q.get(claim_idx, 0.0)
    return covered / total


def criterion_intervals(
    judgment: Judgment,
    claim_q: dict[int, float],
    weights: dict[str, float],
    tau_loc: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Construct verified criterion intervals on ``{-1, 0, +1}``.

    A criterion conclusion constrains the comparison only when its complete
    support chain reaches ``tau_loc``. A verified A/B/tie conclusion induces
    the point interval +1/-1/0; otherwise the criterion remains unresolved at
    ``[-1, 1]``.
    """
    if not 0.0 < tau_loc <= 1.0:
        raise ValueError("tau_loc must be in (0, 1]")
    result_q: dict[str, float] = {}
    winner: dict[str, str] = {}
    for block, indices in zip(judgment.criteria, _criterion_claim_indices(judgment)):
        result_q[block.name] = claim_q.get(indices[-1], 0.0)
        winner[block.name] = block.winner

    direction = {"A": 1.0, "B": -1.0, "tie": 0.0}
    intervals: dict[str, tuple[float, float]] = {}
    for name in weights:
        if name in winner and result_q.get(name, 0.0) >= tau_loc:
            value = direction[winner[name]]
            intervals[name] = (value, value)
        else:
            intervals[name] = (-1.0, 1.0)
    return intervals


def decision_support(
    judgment: Judgment,
    intervals: dict[str, tuple[float, float]],
    weights: dict[str, float],
) -> float:
    """Closed-form conservative support for the certificate's verdict."""
    if judgment.verdict not in ("A", "B"):
        return 0.0
    v = 1.0 if judgment.verdict == "A" else -1.0
    center = 0.0
    uncertainty = 0.0
    for name, weight in weights.items():
        lower, upper = intervals.get(name, (-1.0, 1.0))
        center += weight * (lower + upper) / 2.0
        uncertainty += weight * (upper - lower) / 2.0
    margin = v * center - uncertainty
    return max(0.0, min(1.0, (1.0 + margin) / 2.0))


def q_proc(s_snd: float, s_cov: float, s_dec: float) -> float:
    return (max(s_snd, 0.0) * max(s_cov, 0.0) * max(s_dec, 0.0)) ** (1.0 / 3.0)


def q_total(outcome: int, q_proc_value: float, lam: float = 0.5) -> float:
    """Outcome-dominant total: any O=1 judgment outranks any O=0 judgment."""
    assert outcome in (0, 1)
    assert 0.0 < lam < 1.0
    return (outcome + lam * q_proc_value) / (1.0 + lam)
