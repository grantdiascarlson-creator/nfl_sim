import pandas as pd
import numpy as np


def _nearest_psd_corr(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix onto the nearest valid (PSD) correlation matrix."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals_clipped = np.clip(eigvals, eps, None)
    C_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd


def build_correlation_matrix(
    panel_path: str,
    corr_out_path: str,
    nobs_out_path: str,
    min_pairs: int = 15,
    shrinkage_k: int = 10,
) -> pd.DataFrame:
    """
    Compute the shrunk, PSD-safe cross-cluster correlation matrix.

    For every pair of (own/opp cluster) role-slots:
      - Compute Pearson correlation across all games where both slots were filled
      - Shrink toward 0 proportional to sample size (empirical Bayes-style)
      - Project to nearest PSD matrix via eigenvalue clipping

    Args:
        panel_path:    path to role-slot panel CSV (output of build_role_panel)
        corr_out_path: path to write final correlation matrix CSV
        nobs_out_path: path to write n-observations matrix CSV (for diagnostics)
        min_pairs:     minimum co-observed games before trusting a correlation estimate
        shrinkage_k:   pseudo-count for shrinkage; higher = more conservative

    Returns:
        Correlation matrix as a DataFrame
    """
    print("[4/6] Building correlation matrix...")

    panel = pd.read_csv(panel_path)

    role_cols = [
        c for c in panel.columns
        if (c.startswith('own_') or c.startswith('opp_')) and c != 'opp_team'
    ]
    X = panel[role_cols].copy()
    n = len(role_cols)

    corr_raw = np.full((n, n), np.nan)
    n_obs = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_raw[i, j] = 1.0
                n_obs[i, j] = X.iloc[:, i].notna().sum()
                continue
            xi, xj = X.iloc[:, i], X.iloc[:, j]
            mask = xi.notna() & xj.notna()
            n_pair = mask.sum()
            n_obs[i, j] = n_pair
            if n_pair >= min_pairs:
                corr_raw[i, j] = np.corrcoef(xi[mask], xj[mask])[0, 1]
            else:
                corr_raw[i, j] = 0.0  # insufficient data -> assume independence

    corr_raw = np.nan_to_num(corr_raw, nan=0.0)
    np.fill_diagonal(corr_raw, 1.0)

    # ── Sparsity zeroing ──────────────────────────────────────────────────
    # Many cluster pairs never co-occur (e.g. QB_dual_threat_elite with
    # RB_bellcow_dual on the same team — Lamar never had a bellcow RB).
    # Keeping a noisy near-zero correlation for these pairs creates 18
    # near-zero eigenvalues and a condition number ~5 million, making
    # Cholesky numerically unstable.
    #
    # Zeroing pairs with n_obs < MIN_PAIRS_FOR_NONZERO is more honest
    # (these pairs are independent by assumption, not by measurement) and
    # drops the condition number from ~5M to ~400.
    #
    # Note: zeroing sparse pairs can introduce small negative eigenvalues
    # (we've broken global consistency). The Higham PSD projection below
    # handles those cleanly — small negatives are much better than rank
    # deficiency from near-zero eigenvalues.
    MIN_PAIRS_FOR_NONZERO = 20
    sparse_mask = (n_obs < MIN_PAIRS_FOR_NONZERO)
    np.fill_diagonal(sparse_mask, False)   # always keep diagonal
    corr_raw[sparse_mask] = 0.0

    n_zeroed = sparse_mask.sum() // 2
    print(f"    Zeroed {n_zeroed} sparse pairs (n_obs < {MIN_PAIRS_FOR_NONZERO})")

    # Shrinkage toward 0: lambda -> 1 (full shrinkage) for small n, -> 0 for large n
    shrink_lambda = shrinkage_k / (n_obs + shrinkage_k)
    corr_shrunk = corr_raw * (1 - shrink_lambda)
    np.fill_diagonal(corr_shrunk, 1.0)

    # Project to nearest valid correlation matrix.
    # After sparsity zeroing, small negative eigenvalues appear from broken
    # global consistency. Use eps=0.01 (not the default 1e-6) — eigenvalues
    # below 0.01 represent directions with essentially no data, so raising
    # them to 0.01 is more honest than a near-zero floor.
    # This gives condition number ~400 vs ~5M with eps=1e-6.
    corr_final = _nearest_psd_corr(corr_shrunk, eps=0.01)

    corr_df = pd.DataFrame(corr_final, index=role_cols, columns=role_cols)
    nobs_df = pd.DataFrame(n_obs, index=role_cols, columns=role_cols)

    corr_df.to_csv(corr_out_path)
    nobs_df.to_csv(nobs_out_path)

    eigvals = np.linalg.eigvalsh(corr_final)
    pos_min = eigvals[eigvals > 1e-10].min() if (eigvals > 1e-10).any() else 1e-10
    cond    = eigvals.max() / pos_min
    n_neg   = (eigvals < 0).sum()
    print(f"    Role-slot dimensions:  {n}x{n}")
    print(f"    Min eigenvalue:        {eigvals.min():.2e}")
    print(f"    Negative eigenvalues:  {n_neg}  (corrected by Higham projection)")
    print(f"    Condition number:      {cond:.0f}  (was ~5M before sparsity zeroing)")
    print(f"    Saved correlation matrix to {corr_out_path}")
    print(f"    Saved n-obs matrix to {nobs_out_path}")

    return corr_df
