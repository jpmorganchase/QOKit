###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Chebyshev warm-start, file I/O, and seed angle loading for the QAOA optimizer.

Provides Chebyshev polynomial interpolation/extrapolation (Zhou et al. 2020
INTERP strategy) to warm-start optimization at depth p from lower-depth
results, plus multi-source seed loading from resource files.
"""

import json
import os

import numpy as _np


# ── Chebyshev INTERP extrapolation ──────────────────────────────


def chebyshev_interp(angles_src, p_src, p_tgt, num_coeffs=None):
    """Extrapolate QAOA angles from depth p_src to p_tgt via Chebyshev interpolation.

    Uses the INTERP strategy from Zhou et al. (2020): positions angles at
    t_i = (i + 0.5) / p on [0, 1], fits a Chebyshev polynomial of degree
    p_src - 1, and evaluates at p_tgt positions.

    Parameters
    ----------
    angles_src : array-like
        Source angles of length p_src.
    p_src : int
        Source depth.
    p_tgt : int
        Target depth.
    num_coeffs : int, optional
        Maximum number of Chebyshev coefficients to use. When set, the
        polynomial degree is clamped to min(num_coeffs - 1, p_src - 1).
        Acts as regularization for extrapolation. Default: None (use all).

    Returns
    -------
    ndarray
        Interpolated angles of length p_tgt.
    """
    angles_src = _np.asarray(angles_src, dtype=float)
    if angles_src.shape != (p_src,):
        raise ValueError(f"angles_src shape {angles_src.shape} != ({p_src},)")

    from numpy.polynomial.chebyshev import Chebyshev

    # Sample positions: midpoints of equal-width bins on [0, 1]
    t_src = (_np.arange(p_src) + 0.5) / p_src
    t_tgt = (_np.arange(p_tgt) + 0.5) / p_tgt

    # Fit Chebyshev polynomial of degree p_src - 1, optionally truncated
    deg = max(p_src - 1, 0)
    if num_coeffs is not None:
        deg = min(deg, max(num_coeffs - 1, 0))
    cheb = Chebyshev.fit(t_src, angles_src, deg=deg, domain=[0, 1])

    return cheb(t_tgt)


def chebyshev_extrap(angles_src, p_src, p_tgt):
    """Extrapolate QAOA angles with truncated Chebyshev degree and pinned first element.

    Implements the regularized extrapolation: uses a truncated Chebyshev
    degree of max(3, floor(2*sqrt(p_src-1))) coefficients and pins
    out[0] = angles_src[0], fitting only positions 2..p.

    Parameters
    ----------
    angles_src : array-like
        Source angles of length p_src.
    p_src : int
        Source depth.
    p_tgt : int
        Target depth.

    Returns
    -------
    ndarray
        Extrapolated angles of length p_tgt.
    """
    from math import floor, sqrt

    angles_src = _np.asarray(angles_src, dtype=float)
    if angles_src.shape != (p_src,):
        raise ValueError(f"angles_src shape {angles_src.shape} != ({p_src},)")

    if p_tgt == 1:
        return _np.array([angles_src[0]])

    if p_src == 1:
        out = _np.full(p_tgt, angles_src[0])
        return out

    num_coeffs = max(3, int(floor(2 * sqrt(p_src - 1))))

    tail = chebyshev_interp(angles_src[1:], p_src - 1, p_tgt - 1, num_coeffs=num_coeffs)
    out = _np.empty(p_tgt)
    out[0] = angles_src[0]
    out[1:] = tail
    return out


# ── Scaled linear extrapolation ─────────────────────────────────


def scaled_extrap(all_angles, all_ps, target_p, degree=2, num_ps_fit=3, interp="cubic"):
    """Extrapolate angles to target_p using per-position trend in p.

    For each normalized position x in linspace(0, 1, target_p):
      1. Interpolate each source curve at x (cubic spline by default).
      2. Fit a polynomial of given degree to the last num_ps_fit values.
      3. Extrapolate to target_p.

    Default: cubic spline within-curve interpolation + quadratic
    extrapolation from the last 3 depths.

    Parameters
    ----------
    all_angles : list of array-like
        Angle arrays for each source p, in ascending p order.
    all_ps : array-like of int
        Source p values, same length as all_angles.
    target_p : int
        Target depth.
    degree : int
        Polynomial degree for extrapolation in p (default 2 = quadratic).
    num_ps_fit : int
        Number of most recent p values to use for the fit (default 3).
    interp : str
        Within-curve interpolation: 'linear' or 'cubic' (default).

    Returns
    -------
    ndarray
        Extrapolated angles of length target_p.
    """
    from scipy.interpolate import CubicSpline

    all_ps = _np.asarray(all_ps)
    target_xs = _np.linspace(0, 1, target_p)
    cutoff = max(0, len(all_ps) - num_ps_fit)
    result = _np.empty(target_p)
    for idx, x in enumerate(target_xs):
        vals_at_x = _np.empty(len(all_angles))
        for j, a in enumerate(all_angles):
            a = _np.asarray(a, dtype=float)
            xs = _np.linspace(0, 1, len(a))
            if len(a) < 2 or interp == "linear":
                vals_at_x[j] = float(_np.interp(x, xs, a))
            else:
                vals_at_x[j] = float(CubicSpline(xs, a)(x))
        coeffs = _np.polyfit(all_ps[cutoff:], vals_at_x[cutoff:], degree)
        result[idx] = _np.polyval(coeffs, target_p)
    return result


# ── Chebyshev parameterization for optimizer ────────────────────


def cheb_basis_matrix(n_c, p):
    """Build the (p, n_c) Chebyshev basis matrix J.

    J[i, j] = T_j(t_i) where t_i = (i+0.5)/p mapped to [0, 1].
    Then: angles = J @ coeffs, and grad_coeffs = J.T @ grad_angles.
    """
    from numpy.polynomial.chebyshev import Chebyshev

    t = (_np.arange(p) + 0.5) / p
    J = _np.empty((p, n_c))
    for j in range(n_c):
        c = _np.zeros(n_c)
        c[j] = 1.0
        J[:, j] = Chebyshev(c, domain=[0, 1])(t)
    return J


def cheb_to_angles(coeffs, p):
    """Reconstruct p angles from n_c Chebyshev coefficients.

    Uses the same sample positions as chebyshev_interp: t_i = (i+0.5)/p on [0,1].

    Parameters
    ----------
    coeffs : array-like
        Chebyshev coefficients of length n_c.
    p : int
        Target number of angles.

    Returns
    -------
    ndarray
        Reconstructed angles of length p.
    """
    from numpy.polynomial.chebyshev import Chebyshev

    coeffs = _np.asarray(coeffs, dtype=float)
    cheb = Chebyshev(coeffs, domain=[0, 1])
    t = (_np.arange(p) + 0.5) / p
    return cheb(t)


def angles_to_cheb(angles, n_c):
    """Project p angles into n_c Chebyshev coefficients.

    Fits a Chebyshev polynomial of degree n_c-1 to the angles at
    positions t_i = (i+0.5)/p on [0,1].

    Parameters
    ----------
    angles : array-like
        Angles of length p.
    n_c : int
        Number of Chebyshev coefficients.

    Returns
    -------
    ndarray
        Chebyshev coefficients of length n_c.
    """
    from numpy.polynomial.chebyshev import Chebyshev

    angles = _np.asarray(angles, dtype=float)
    p = len(angles)
    t = (_np.arange(p) + 0.5) / p
    deg = min(n_c - 1, p - 1)
    cheb = Chebyshev.fit(t, angles, deg=deg, domain=[0, 1])
    c = cheb.coef
    if len(c) < n_c:
        c = _np.concatenate([c, _np.zeros(n_c - len(c))])
    return c[:n_c]


# ── Output file I/O ─────────────────────────────────────────────


def load_output_file(path):
    """Load an optimizer output JSON file.

    Returns the parsed dict, or None if the file does not exist.
    """
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return None
    return json.loads(content)


def save_output_file(path, k, D, results_dict):
    """Save or merge optimizer results into a JSON file.

    Parameters
    ----------
    path : str
        Output file path.
    k, D : int
        Problem parameters. Must match any existing file.
    results_dict : dict
        Mapping from str(p) -> result dict for each depth.

    Raises
    ------
    ValueError
        If the existing file has different k or D.
    """
    existing = load_output_file(path)
    if existing is not None:
        if existing.get("k") != k or existing.get("D") != D:
            raise ValueError(f"Output file {path} has k={existing.get('k')}, D={existing.get('D')} " f"but requested k={k}, D={D}")
        existing["results"].update(results_dict)
        data = existing
    else:
        data = {
            "k": k,
            "D": D,
            "convention": "e^{-ig Z^k}",
            "results": results_dict,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Seed angle loading ──────────────────────────────────────────

_RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)


def _check_results_data(data, k, D, p_target, extrap=True):
    """Check parsed results JSON for seed angles.

    Parameters
    ----------
    extrap : bool
        If True (default), also try extrapolation from the largest p < p_target.
        If False, only return exact p_target matches.

    Returns (gammas, betas, suffix_str) or None.
    suffix_str is e.g. "p=5" for exact match or "p=3->extrap" for extrapolation.
    """
    if data is None or data.get("k") != k or data.get("D") != D:
        return None
    results = data.get("results", {})

    if str(p_target) in results:
        entry = results[str(p_target)]
        if entry.get("converged", True):
            return (
                _np.array(entry["gammas"]),
                _np.array(entry["betas"]),
                f"p={p_target}",
            )

    if not extrap:
        return None

    available = sorted(
        [int(p) for p in results if int(p) < p_target and results[p].get("converged", True)],
    )
    if len(available) >= 3:
        # Scaled cubic + quadratic extrapolation (3+ source depths)
        all_g = [results[str(p)]["gammas"] for p in available]
        all_b = [results[str(p)]["betas"] for p in available]
        gammas = scaled_extrap(all_g, available, p_target)
        betas = scaled_extrap(all_b, available, p_target)
        return gammas, betas, f"p={available[-2]},{available[-1]}->scaled"
    elif len(available) == 2:
        # Scaled linear extrapolation (only 2 source depths)
        all_g = [results[str(p)]["gammas"] for p in available]
        all_b = [results[str(p)]["betas"] for p in available]
        gammas = scaled_extrap(all_g, available, p_target, degree=1, num_ps_fit=2, interp="linear")
        betas = scaled_extrap(all_b, available, p_target, degree=1, num_ps_fit=2, interp="linear")
        return gammas, betas, f"p={available[-2]},{available[-1]}->scaled"
    elif available:
        # Fallback: single source, Chebyshev extrapolation
        p_src = available[0]
        entry = results[str(p_src)]
        gammas = chebyshev_extrap(entry["gammas"], p_src, p_target)
        betas = chebyshev_extrap(entry["betas"], p_src, p_target)
        return gammas, betas, f"p={p_src}->extrap"

    return None


def _try_load_D_specific(k, D, p):
    """Try to load angles from D-specific resource files."""
    if k != 2:
        return None

    if D == 3:
        path = os.path.join(_RESOURCES_DIR, "angles_k2_D3.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                entry = data.get(str(p))
                if entry is not None:
                    return -_np.array(entry["gamma"]), _np.array(entry["beta"]), "angles_k2_D3"
            except Exception:
                pass

    path = os.path.join(_RESOURCES_DIR, "angles_k2.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            for entry in data.values():
                if entry["d"] == D and entry["p"] == p:
                    return -_np.array(entry["gamma"]) / 4, _np.array(entry["beta"]), "angles_k2"
        except Exception:
            pass

    return None


def _discover_nearby_results(k, D):
    """Find results_k*_D*.json files with k' <= k and D' <= D."""
    import re

    pattern = re.compile(r"^results_k(\d+)_D(\d+)\.json$")
    candidates = []
    try:
        for name in os.listdir(_RESOURCES_DIR):
            m = pattern.match(name)
            if m:
                k2, D2 = int(m.group(1)), int(m.group(2))
                if k2 <= k and D2 <= D and (k2, D2) != (k, D):
                    candidates.append((k2, D2, name))
    except OSError:
        pass
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    return candidates


def _try_load_Dinf(k, p, D=None):
    """Try to load angles from angles_Dinf.json."""
    path = os.path.join(_RESOURCES_DIR, "angles_Dinf.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        ks, ps = str(k), str(p)
        if ks in data and ps in data[ks]:
            entry = data[ks][ps]
            gammas = -_np.array(entry["gammas"])
            betas = _np.array(entry["betas"])
            if D is not None:
                gammas = gammas / _np.sqrt(D)
            return gammas, betas
    except Exception:
        pass
    return None


def load_seed_angles(k, D, p_target, output_file=None):
    """Find seed angles for optimization at depth p_target.

    Search priority:
    1. Output file: exact match at p_target
    2. Auto-discovered results_k{k}_D{D}.json: exact match
    3. Output file: largest p < p_target -> extrap
    4. Auto-discovered results_k{k}_D{D}.json: largest p < p_target -> extrap
    5. D->inf angles (-gamma/sqrt(D)): exact match at p_target
    6. D->inf angles (-gamma/sqrt(D)): descending p_target-1,...,1 -> extrap
    7. D-specific angle files: exact match at p_target
    8. D-specific angle files: descending p_target-1,...,1 -> extrap
    9-10. Nearby results files (k' <= k, D' <= D, closest first)
    11. Heuristic linspace fallback

    Parameters
    ----------
    k, D : int
        Problem parameters.
    p_target : int
        Target QAOA depth.
    output_file : str, optional
        Path to optimizer output file.

    Returns
    -------
    gammas : ndarray
        Seed gamma angles of length p_target.
    betas : ndarray
        Seed beta angles of length p_target.
    source : str
        Description of where the seed came from.
    """
    # 1-2. Check output file (exact match + extrap)
    res = _check_results_data(load_output_file(output_file), k, D, p_target)
    if res is not None:
        fname = os.path.basename(output_file)
        return res[0], res[1], f"output_file:{fname}({res[2]})"

    # 3-4. Auto-discovered results_k{k}_D{D}.json (exact match + extrap)
    results_name = f"results_k{k}_D{D}.json"
    results_path = os.path.join(_RESOURCES_DIR, results_name)
    skip_results = output_file is not None and os.path.abspath(output_file) == os.path.abspath(results_path)
    if not skip_results:
        res = _check_results_data(load_output_file(results_path), k, D, p_target)
        if res is not None:
            return res[0], res[1], f"results:{results_name}({res[2]})"

    # 5. D->inf angles (-gamma/sqrt(D)): exact match
    res = _try_load_Dinf(k, p_target, D=D)
    if res is not None:
        return res[0], res[1], f"angles_Dinf(p={p_target},g/sqrt({D}))"

    # 6. D->inf angles (-gamma/sqrt(D)): descend from p_target-1
    for p_src in range(p_target - 1, 0, -1):
        res = _try_load_Dinf(k, p_src, D=D)
        if res is not None:
            gammas = chebyshev_extrap(res[0], p_src, p_target)
            betas = chebyshev_extrap(res[1], p_src, p_target)
            return gammas, betas, f"angles_Dinf(p={p_src},g/sqrt({D}))->extrap"

    # 7. D-specific angle files: exact match
    res = _try_load_D_specific(k, D, p_target)
    if res is not None:
        gammas, betas, tag = res
        return gammas, betas, f"{tag}(p={p_target})"

    # 8. D-specific angle files: descend from p_target-1
    for p_src in range(p_target - 1, 0, -1):
        res = _try_load_D_specific(k, D, p_src)
        if res is not None:
            gammas_src, betas_src, tag = res
            gammas = chebyshev_extrap(gammas_src, p_src, p_target)
            betas = chebyshev_extrap(betas_src, p_src, p_target)
            return gammas, betas, f"{tag}(p={p_src})->extrap"

    # 9-10. Nearby results files (k' <= k, D' <= D, closest first)
    for k2, D2, fname in _discover_nearby_results(k, D):
        fpath = os.path.join(_RESOURCES_DIR, fname)
        res = _check_results_data(load_output_file(fpath), k2, D2, p_target)
        if res is not None:
            return res[0], res[1], f"results:{fname}({res[2]})"

    # 11. Heuristic fallback
    gammas = _np.linspace(0.1, 0.5, p_target)
    betas = _np.linspace(0.5, 0.1, p_target)
    return gammas, betas, "heuristic"
