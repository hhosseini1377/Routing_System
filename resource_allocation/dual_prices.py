from __future__ import annotations

import numpy as np

def fractions_to_counts(w: np.ndarray, N: int) -> np.ndarray:
    """
    Convert fractions w (sum to 1) into integer counts c that sum to N
    using largest-remainder rounding.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("w must be 1D")
    if not np.isfinite(w).all():
        raise ValueError("w contains non-finite values")
    if (w < 0).any():
        raise ValueError("w must be nonnegative")
    s = w.sum()
    if not np.isclose(s, 1.0):
        w = w / s

    raw = w * N
    c = np.floor(raw).astype(int)
    r = N - c.sum()
    if r > 0:
        rema = raw - c
        idx = np.argsort(-rema)[:r]  # largest remainders
        c[idx] += 1
    elif r < 0:
        # If numerical issues cause sum > N, remove from smallest remainders
        rema = raw - c
        idx = np.argsort(rema)[:(-r)]
        c[idx] -= 1

    # Safety
    if c.sum() != N:
        raise RuntimeError("Rounding failed to produce counts summing to N")
    if (c < 0).any():
        raise RuntimeError("Rounding produced negative counts")
    return c


def _repair_to_exact_counts(S: np.ndarray, assign: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Greedy repair to enforce exact counts.
    Moves prompts from overfull models to underfull models with minimal score loss.
    """
    N, K = S.shape
    counts = np.bincount(assign, minlength=K).astype(int)

    # If already exact, nothing to do
    if np.all(counts == c):
        return assign

    # Lists of indices currently assigned to each model
    buckets = [np.where(assign == i)[0].tolist() for i in range(K)]

    # We will move one prompt at a time until counts match.
    # Since K is small (3), this is fast enough in practice.
    while True:
        over = np.where(counts > c)[0]
        under = np.where(counts < c)[0]
        if len(over) == 0 and len(under) == 0:
            break

        # Pick the best single move among all (over_model -> under_model) pairs
        best_move = None  # (loss, j, a, b)
        for a in over:
            if not buckets[a]:
                continue
            idx_a = np.array(buckets[a], dtype=int)
            s_a = S[idx_a, a]  # current scores in model a

            for b in under:
                # loss if move j from a to b: s_a - s_b
                losses = s_a - S[idx_a, b]
                m = int(np.argmin(losses))
                loss = float(losses[m])
                j = int(idx_a[m])
                if (best_move is None) or (loss < best_move[0]):
                    best_move = (loss, j, int(a), int(b))

        if best_move is None:
            raise RuntimeError("Repair failed: no feasible moves found")

        _, j, a, b = best_move

        # Apply move
        assign[j] = b
        counts[a] -= 1
        counts[b] += 1

        # Update buckets lists (remove j from a, add to b)
        # O(len) removal, but K small; still fine for N~36k.
        buckets[a].remove(j)
        buckets[b].append(j)

    # Final check
    counts2 = np.bincount(assign, minlength=K)
    if not np.all(counts2 == c):
        raise RuntimeError(f"Repair did not reach exact counts. got={counts2}, want={c}")
    return assign


def score_under_fractions_dual(
    S: np.ndarray,
    w: np.ndarray,
    max_iter: int = 2000,
    eta0: float = 1e-4,
    tol: int = 0,
    tie_noise: float = 1e-9,
    seed: int = 0,
    do_repair: bool = True,
    alpha_init: np.ndarray | None = None,
):
    """
    Compute S_hat(w) via dual 'prices' method:
      - Maintain prices alpha_i
      - Assign each prompt to argmax_i (s_ji - alpha_i)
      - Update alpha to push counts toward target c_i

    Parameters
    ----------
    S : (N,K) array
        Predicted scores s_ji
    w : (K,) array
        Routing fractions, should sum to 1 (will be normalized if not)
    max_iter : int
        Number of dual iterations
    eta0 : float
        Base step size; actual eta_t = eta0 / sqrt(t+1)
    tol : int
        Stop if max |count_i - c_i| <= tol
    tie_noise : float
        Tiny noise added to break ties deterministically
    seed : int
        RNG seed for tie noise
    do_repair : bool
        If True, enforce exact counts via greedy repair at the end.

    Returns
    -------
    S_hat : float
        Achieved average score under exact fractions
    assign : (N,) int array
        Model index per prompt (0..K-1)
    counts : (K,) int array
        Prompts per model (matches c if do_repair=True)
    c : (K,) int array
        Target counts derived from w
    alpha : (K,) float array
        Final learned prices
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be 2D (N,K)")
    N, K = S.shape
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != K:
        raise ValueError(f"w must have length K={K}, got {w.shape[0]}")

    c = fractions_to_counts(w, N)

    rng = np.random.default_rng(seed)
    if alpha_init is None:
        alpha = np.zeros(K, dtype=float)
    else:
        alpha = np.asarray(alpha_init, dtype=float).reshape(K)

    # Pre-generate tiny tie-breaking noise (fixed across iters)
    noise = tie_noise * rng.standard_normal(size=S.shape)

    assign = np.zeros(N, dtype=int)
    # print('..........................................................')
    for t in range(max_iter):
        eta = eta0 / np.sqrt(t + 1.0)

        # Adjusted scores: s_ji - alpha_i
        # broadcast alpha over rows
        adjusted = S - alpha[None, :] + noise

        assign = np.argmax(adjusted, axis=1).astype(int)
        counts = np.bincount(assign, minlength=K).astype(int)

        diff = counts - c
        
        # print(t, diff  )
        if np.max(np.abs(diff)) <= tol:
            break

        # Subgradient ascent on dual (equivalently: increase price if overfull)
        alpha = alpha + eta * diff

        # Optional: keep alpha centered (helps numerics; doesn't change argmax decisions)
        alpha = alpha - alpha.mean()

    # Make counts exact if needed
    if do_repair:
        assign = _repair_to_exact_counts(S, assign.copy(), c)
        counts = np.bincount(assign, minlength=K).astype(int)
    else:
        counts = np.bincount(assign, minlength=K).astype(int)

    S_hat = float(np.mean(S[np.arange(N), assign]))
    return S_hat, assign, counts, c, alpha