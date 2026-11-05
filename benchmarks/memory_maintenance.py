"""RQ2: bounded memory-maintenance poli
cies for a labelled embedding stream.

Every policy holds exactly K slots per class (pattern m and count n); capacity
never changes. Similarities use the centred cosine of retrieval with a fixed
centre mu. Policies:

    frozen     ignore arrivals (the repository's current behaviour)
    fifo       overwrite the oldest slot of the arrival's class
    reservoir  class-balanced reservoir sampling of real embeddings
    fl_swap    swap the arrival into the slot with the largest positive
               facility-location coverage gain on a bounded reference set
    ra         running average: nearest same-class slot m <- m + (z - m) / (n + 1)
    ra_fl      ra plus a periodic layout check: merge the two most similar slots
               and spawn the reference point with the largest coverage gain,
               only if coverage on the reference set increases

The per-class reference set is a bounded reservoir of seen embeddings, used
only to measure coverage (never retrieved from).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_EPS = 1e-8


def unit(z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Centred, L2-normalised float64 vectors (zero-norm -> zero, as in retrieval)."""
    return F.normalize(z.double() - mu.double(), p=2, dim=-1, eps=_EPS)


def stream_split(labels: torch.Tensor, n0: int, per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(initial rows, stream rows) from one seeded order of all rows.

    Initial = the first n0 rows of each class in that order; stream = the next
    per_class rows of each class, interleaved in that same order.
    """
    order = torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(seed))
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel())
    init, stream = [], []
    for c in labels.unique(sorted=True).tolist():
        rows = order[labels[order] == c]
        if rows.numel() < n0 + per_class:
            raise ValueError(f"class {c} has {rows.numel()} rows, fewer than n0 + per_class = {n0 + per_class}.")
        init.append(rows[:n0])
        stream.append(rows[n0:n0 + per_class])
    stream_rows = torch.cat(stream)
    return torch.cat(init).sort().values, stream_rows[position[stream_rows].argsort()]


class Policy:
    """Base class: K slots per class, reference reservoirs, counters, optional per-arrival log."""

    name = "base"

    def __init__(
        self,
        patterns: torch.Tensor,
        counts: torch.Tensor,
        mu: torch.Tensor,
        init_z: torch.Tensor,
        init_y: torch.Tensor,
        *,
        seed: int,
        ref_size: int = 200,
        layout_every: int = 50,
        epsilon: float = 0.0,
        log: bool = False,
    ) -> None:
        if patterns.ndim != 3 or counts.shape != patterns.shape[:2]:
            raise ValueError("patterns must be [C, K, D] and counts [C, K].")
        self.P = patterns.double().clone()
        self.n = counts.double().clamp_min(1.0).clone()
        self.C, self.K, self.D = self.P.shape
        self.mu = mu.double()
        self.U = unit(self.P, self.mu)
        self.gen = torch.Generator().manual_seed(seed)
        self.ref_size, self.layout_every, self.epsilon = ref_size, layout_every, epsilon
        self.ref_raw: list[torch.Tensor] = []
        self.ref_unit: list[torch.Tensor] = []
        self.ref_seen = torch.zeros(self.C, dtype=torch.long)
        self.arrivals = torch.zeros(self.C, dtype=torch.long)
        self.seen = torch.bincount(init_y, minlength=self.C).long()
        for c in range(self.C):
            zc = init_z[init_y == c].double()
            if zc.shape[0] > ref_size:
                zc = zc[torch.randperm(zc.shape[0], generator=self.gen)[:ref_size]]
            self.ref_raw.append(zc.clone())
            self.ref_unit.append(unit(zc, self.mu))
            self.ref_seen[c] = int((init_y == c).sum())
        self.stats = {"arrivals": 0, "changed": 0, "swaps": 0, "absorbed": 0,
                      "layout_checks": 0, "layout_changes": 0, "conflicts": 0}
        self.log: list[dict[str, Any]] | None = [] if log else None

    # -- helpers -----------------------------------------------------------
    def _set(self, c: int, j: int, z: torch.Tensor, count: float) -> None:
        self.P[c, j] = z
        self.n[c, j] = count
        self.U[c, j] = unit(z, self.mu)

    def _update_reference(self, c: int, z: torch.Tensor, u: torch.Tensor) -> None:
        self.ref_seen[c] += 1
        if self.ref_raw[c].shape[0] < self.ref_size:
            self.ref_raw[c] = torch.cat([self.ref_raw[c], z[None]])
            self.ref_unit[c] = torch.cat([self.ref_unit[c], u[None]])
            return
        r = int(torch.randint(0, int(self.ref_seen[c]), (1,), generator=self.gen))
        if r < self.ref_size:
            self.ref_raw[c][r] = z
            self.ref_unit[c][r] = u

    def coverage(self, c: int, U_c: torch.Tensor | None = None) -> float:
        """Facility-location objective on class c's reference set: sum_r max_j q(r, m_j)."""
        U_c = self.U[c] if U_c is None else U_c
        return float(((1.0 + self.ref_unit[c] @ U_c.T) / 2.0).max(dim=1).values.sum())

    def check_invariants(self) -> None:
        assert self.P.shape == (self.C, self.K, self.D), "capacity changed"
        assert torch.isfinite(self.P).all() and torch.isfinite(self.n).all(), "non-finite memory"
        assert (self.n >= 1).all(), "slot count below 1"

    # -- public API ----------------------------------------------------------
    def observe(self, z: torch.Tensor, y: int) -> dict[str, Any]:
        z = z.double()
        u = unit(z, self.mu)
        nearest = int(torch.argmax(self.U.reshape(-1, self.D) @ u)) // self.K
        conflict = nearest != y
        self.stats["arrivals"] += 1
        self.stats["conflicts"] += int(conflict)
        self.arrivals[y] += 1
        self.seen[y] += 1
        decision = self._update(y, z, u)
        decision.update({"t": self.stats["arrivals"], "class": y, "conflict": conflict})
        if decision["action"] != "keep":
            self.stats["changed"] += 1
        self._update_reference(y, z, u)
        if self.log is not None:
            self.log.append(decision)
        return decision

    def _update(self, c: int, z: torch.Tensor, u: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError

    def memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(patterns [C*K, D] float32, labels [C*K]) in class-block order."""
        labels = torch.arange(self.C).repeat_interleave(self.K)
        return self.P.reshape(-1, self.D).float().clone(), labels

    def extra_state_floats(self) -> int:
        return 0


class Frozen(Policy):
    name = "frozen"

    def _update(self, c, z, u):
        return {"action": "keep", "slot": -1, "delta": 0.0}


class FIFO(Policy):
    name = "fifo"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ptr = torch.zeros(self.C, dtype=torch.long)

    def _update(self, c, z, u):
        j = int(self.ptr[c])
        self._set(c, j, z, 1.0)
        self.ptr[c] = (j + 1) % self.K
        self.stats["swaps"] += 1
        return {"action": "replace", "slot": j, "delta": 0.0}

    def extra_state_floats(self) -> int:
        return self.C


class Reservoir(Policy):
    name = "reservoir"

    def _update(self, c, z, u):
        r = int(torch.randint(0, int(self.seen[c]), (1,), generator=self.gen))
        if r < self.K:
            self._set(c, r, z, 1.0)
            self.stats["swaps"] += 1
            return {"action": "replace", "slot": r, "delta": 0.0}
        return {"action": "keep", "slot": -1, "delta": 0.0}

    def extra_state_floats(self) -> int:
        return self.C


class FLSwap(Policy):
    name = "fl_swap"

    def swap_gains(self, c: int, u: torch.Tensor) -> torch.Tensor:
        """Delta_j = F((M \\ m_j) + z) - F(M) on the reference set, for every slot j."""
        R = self.ref_unit[c]
        Q = (1.0 + R @ self.U[c].T) / 2.0                       # [r, K]
        qz = (1.0 + R @ u) / 2.0                                 # [r]
        top = Q.topk(min(2, self.K), dim=1)
        best1, arg1 = top.values[:, 0], top.indices[:, 0]
        best2 = top.values[:, 1] if self.K > 1 else torch.zeros_like(best1)
        is_best = arg1[:, None] == torch.arange(self.K)[None, :]
        others = torch.where(is_best, best2[:, None], best1[:, None])   # coverage without slot j
        return torch.maximum(others, qz[:, None]).sum(dim=0) - best1.sum()

    def _update(self, c, z, u):
        gains = self.swap_gains(c, u)
        j = int(torch.argmax(gains))
        delta = float(gains[j])
        if delta > self.epsilon:
            self._set(c, j, z, 1.0)
            self.stats["swaps"] += 1
            return {"action": "replace", "slot": j, "delta": delta}
        return {"action": "keep", "slot": -1, "delta": delta}

    def extra_state_floats(self) -> int:
        return self.C * self.ref_size * self.D * 2


class RunningAverage(Policy):
    name = "ra"
    layout = False

    def _update(self, c, z, u):
        j = int(torch.argmax(self.U[c] @ u))
        n = float(self.n[c, j])
        self._set(c, j, self.P[c, j] + (z - self.P[c, j]) / (n + 1.0), n + 1.0)
        self.stats["absorbed"] += 1
        decision = {"action": "absorb", "slot": j, "delta": 0.0}
        if self.layout and int(self.arrivals[c]) % self.layout_every == 0:
            change = self.layout_step(c)
            if change is not None:
                decision.update(change)
        return decision

    def layout_step(self, c: int) -> dict[str, Any] | None:
        """Merge the two most similar slots and spawn the best-covering reference point, if coverage rises."""
        self.stats["layout_checks"] += 1
        R, U_c = self.ref_unit[c], self.U[c]
        if R.shape[0] == 0 or self.K < 2:
            return None
        best = ((1.0 + R @ U_c.T) / 2.0).max(dim=1).values
        before = float(best.sum())
        gain = ((1.0 + R @ R.T) / 2.0 - best[:, None]).clamp_min(0.0).sum(dim=0)
        i = int(torch.argmax(gain))
        sim = U_c @ U_c.T
        sim.fill_diagonal_(-float("inf"))
        a, b = divmod(int(torch.argmax(sim)), self.K)
        na, nb = float(self.n[c, a]), float(self.n[c, b])
        merged = (na * self.P[c, a] + nb * self.P[c, b]) / (na + nb)
        U_new = U_c.clone()
        U_new[a] = unit(merged, self.mu)
        U_new[b] = R[i]
        after = self.coverage(c, U_new)
        if after - before > self.epsilon:
            self._set(c, a, merged, na + nb)
            self._set(c, b, self.ref_raw[c][i].clone(), 1.0)
            self.stats["layout_changes"] += 1
            return {"layout": "merge_spawn", "merged": [a, b], "spawn_slot": b, "layout_delta": after - before}
        return None

    def extra_state_floats(self) -> int:
        return self.C * self.K + (self.C * self.ref_size * self.D * 2 if self.layout else 0)


class RunningAverageLayout(RunningAverage):
    name = "ra_fl"
    layout = True


POLICY_CLASSES = {cls.name: cls for cls in (Frozen, FIFO, Reservoir, FLSwap, RunningAverage, RunningAverageLayout)}
