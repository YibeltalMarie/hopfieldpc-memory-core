"""Memory construction by selecting real training embeddings (research question RQ1).

The existing benchmark stores class-wise k-means centroids, which are synthetic
averages. This module instead stores actual training embeddings, chosen class
by class:

* ``facility_location`` (component C1): greedy maximisation of the coverage
  objective

      F(S) = sum_{z_i in R} max_{m in S} q(z_i, m),
      q(z, m) = (1 + cos(z - mu, m - mu)) / 2,

  where R is the class's reference set (here: all of its training embeddings)
  and mu is the same centre the Hopfield bridge uses for retrieval scores, so
  selection and retrieval share one geometry.
* ``random``: uniform sampling without replacement, the control that any
  structured selector has to beat.

``q`` is an increasing affine map of cosine, so greedy selection under ``q`` is
identical to greedy selection under raw cosine; the shift only makes the
similarities nonnegative, as monotone facility location requires.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SELECTION_METHODS = ("kmeans", "facility_location", "random")

# Same epsilon as hopfieldpc.functional._normalize_for_scores: a zero-norm
# centred vector normalises to zero, i.e. cosine 0, exactly as in retrieval.
_NORM_EPS = 1e-8


def coverage_similarity(
    reference: torch.Tensor,
    candidates: torch.Tensor,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """q(z, m) = (1 + cos(z - mu, m - mu)) / 2 in [0, 1].

    reference [N_ref, D], candidates [N_cand, D], center [D] or None (no centring).
    Returns [N_ref, N_cand] in float64 so greedy sums and tie-breaks are stable.
    """
    if reference.ndim != 2 or candidates.ndim != 2 or reference.shape[1] != candidates.shape[1]:
        raise ValueError(
            f"reference and candidates must be [N, D] with equal D, got "
            f"{tuple(reference.shape)} and {tuple(candidates.shape)}."
        )
    ref = reference.double()
    cand = candidates.double()
    if center is not None:
        if center.shape != (reference.shape[1],):
            raise ValueError(f"center must be [{reference.shape[1]}], got {tuple(center.shape)}.")
        mu = center.double()
        ref, cand = ref - mu, cand - mu
    ref = F.normalize(ref, p=2, dim=1, eps=_NORM_EPS)
    cand = F.normalize(cand, p=2, dim=1, eps=_NORM_EPS)
    q = (1.0 + ref @ cand.T) / 2.0
    if not torch.isfinite(q).all():
        raise ValueError("coverage similarity contains NaN or inf; check the embeddings.")
    return q.clamp_(0.0, 1.0)


def facility_location_objective(q: torch.Tensor, selected: torch.Tensor | list[int]) -> float:
    """F(S) = sum_i max_{j in S} q[i, j]; F(empty set) = 0."""
    idx = torch.as_tensor(selected, dtype=torch.long)
    if idx.numel() == 0:
        return 0.0
    return float(q[:, idx].max(dim=1).values.sum().item())


def greedy_facility_location(
    q: torch.Tensor,
    k: int,
    *,
    eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily pick k candidate columns of q [N_ref, N_cand] maximising F.

    Marginal gain of candidate j given the current set S:
        gain_j = sum_i max(0, q[i, j] - best_i),  best_i = max_{m in S} q[i, m].
    Each step adds the eligible, unselected candidate with the largest gain;
    exact ties go to the lowest index (deterministic). ``eligible`` [N_cand]
    bool restricts who may be selected; every reference row is still covered.

    Returns (indices [k] in selection order, gains [k]).
    """
    if q.ndim != 2:
        raise ValueError(f"q must be [N_ref, N_cand], got {tuple(q.shape)}.")
    if (q < 0).any():
        raise ValueError("facility location requires nonnegative similarities.")
    n_ref, n_cand = q.shape
    if eligible is None:
        eligible = torch.ones(n_cand, dtype=torch.bool, device=q.device)
    if eligible.shape != (n_cand,) or eligible.dtype != torch.bool:
        raise ValueError(f"eligible must be a bool tensor of shape [{n_cand}].")
    if not 1 <= k <= int(eligible.sum().item()):
        raise ValueError(f"k ({k}) must be between 1 and the number of eligible candidates "
                         f"({int(eligible.sum().item())}).")

    best = torch.zeros(n_ref, dtype=q.dtype, device=q.device)
    blocked = (~eligible).to(q.device)
    indices, gains = [], []
    for _ in range(k):
        gain = (q - best[:, None]).clamp_(min=0.0).sum(dim=0)
        gain[blocked] = -torch.inf
        j = int(torch.argmax(gain).item())  # first maximum -> lowest index on ties
        indices.append(j)
        gains.append(float(gain[j].item()))
        best = torch.maximum(best, q[:, j])
        blocked[j] = True
    return torch.tensor(indices, dtype=torch.long), torch.tensor(gains, dtype=torch.float64)


def random_selection(n: int, k: int, *, generator: torch.Generator) -> torch.Tensor:
    """k distinct indices drawn uniformly from range(n)."""
    if not 1 <= k <= n:
        raise ValueError(f"k ({k}) must be between 1 and n ({n}).")
    return torch.randperm(n, generator=generator)[:k]


def select_class_memory(
    z_train: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    method: str,
    center: torch.Tensor | None,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-wise selection of real embeddings.

    For each class c, candidates and reference set are that class's training
    embeddings. Returns (patterns [C*per_class, D] in z_train's dtype,
    pattern labels [C*per_class], source row indices into z_train [C*per_class]).
    """
    if method not in ("facility_location", "random"):
        raise ValueError(f"unknown selection method {method!r}; "
                         "expected 'facility_location' or 'random'.")
    generator = torch.Generator().manual_seed(seed)
    rows, block_labels = [], []
    for c in range(num_classes):
        class_rows = torch.nonzero(labels == c, as_tuple=True)[0]
        if class_rows.numel() < per_class:
            raise ValueError(f"class {c} has {class_rows.numel()} embeddings, "
                             f"fewer than per_class={per_class}.")
        if method == "facility_location":
            zc = z_train[class_rows]
            local, _ = greedy_facility_location(coverage_similarity(zc, zc, center), per_class)
        else:
            local = random_selection(class_rows.numel(), per_class, generator=generator)
        rows.append(class_rows[local])
        block_labels.append(torch.full((per_class,), c, dtype=torch.long))
    source = torch.cat(rows)
    return z_train[source].clone(), torch.cat(block_labels), source
