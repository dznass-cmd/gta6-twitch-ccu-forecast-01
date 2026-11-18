"""gta6-twitch-ccu-forecast-01: Algorix Weighted Block Ridge Quantile Forecaster.

Predicts Twitch CCU quantiles [0.05, 0.25, 0.50, 0.75, 0.95] for launch day (hours 2..24).
Optimized for minimum pinball loss under Python 3.13 standard library.
"""

from datetime import date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LOG_V1_MIN = math.log(15000.0)
LOG_V1_MAX = math.log(400000.0)
RIDGE_LAMBDA = 0.8


def clip(val: float, lo: float, hi: float) -> float:
    """Clamps float val into [lo, hi]."""
    return lo if val < lo else (hi if val > hi else val)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def clamp_bounds(val: float | int) -> float | int:
    """Clamps forecast CCU to valid non-negative range [0.01, 1e9]."""
    return min(1e9, max(0.01, val))


def is_us_edt(day_str: str) -> float:
    """Returns 1.0 when day_str (YYYY-MM-DD) falls inside US Daylight Saving Time, else 0.0."""
    year, month, day = map(int, day_str[:10].split("-"))
    dt = date(year, month, day)
    mar1 = date(year, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = date(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return 1.0 if dst_start <= dt < dst_end else 0.0


def extract_pre_features(
    hourly_map: dict[str, dict], day_start_dt: datetime, v1_val: float
) -> tuple[float, float]:
    """Extracts pre-checkpoint geometric-mean log ratios (log_dip for t=-6..-3, log_late for t=-2..-1)."""
    eve_logs = []
    for offset in range(-6, -2):
        key = (day_start_dt + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pt = hourly_map.get(key)
        if pt is not None:
            eve_logs.append(math.log(max(1.0, float(pt.get("viewers", 1)))))
    v_eve = math.exp(sum(eve_logs) / len(eve_logs)) if eve_logs else v1_val

    late_logs = []
    for offset in range(-2, 0):
        key = (day_start_dt + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pt = hourly_map.get(key)
        if pt is not None:
            late_logs.append(math.log(max(1.0, float(pt.get("viewers", 1)))))
    v_late = math.exp(sum(late_logs) / len(late_logs)) if late_logs else v1_val

    log_dip = clip(math.log(v_eve / v1_val), -0.8, 1.5)
    log_late = clip(math.log(v_late / v1_val), -0.7, 0.8)
    return log_dip, log_late


def weighted_quantile(vals: list[float], weights: list[float], p: float) -> float:
    """Computes weighted quantile p via linear interpolation at cumulative weight midpoints."""
    pairs = sorted(zip(vals, weights), key=lambda pair: pair[0])
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        return pairs[len(pairs) // 2][0]

    mids = []
    cum = 0.0
    for _, w in pairs:
        mids.append((cum + 0.5 * w) / total)
        cum += w

    if p <= mids[0]:
        return pairs[0][0]
    if p >= mids[-1]:
        return pairs[-1][0]

    for idx in range(len(pairs) - 1):
        if mids[idx] <= p <= mids[idx + 1]:
            span = mids[idx + 1] - mids[idx]
            frac = 0.0 if span <= 0.0 else (p - mids[idx]) / span
            return pairs[idx][0] + frac * (pairs[idx + 1][0] - pairs[idx][0])
    return pairs[-1][0]


def invert_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Inverts a small square matrix via Gauss-Jordan elimination with partial pivoting."""
    n = len(mat)
    aug = [mat[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            aug[col][col] = 1e-12
        inv_piv = 1.0 / aug[col][col]
        for j in range(col, 2 * n):
            aug[col][j] *= inv_piv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                for j in range(col, 2 * n):
                    aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def select_comparables(case: dict, v1: float) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Filters and weights comparables and returns target feature tuple."""
    ds_target = str(case.get("day_start", ""))[:10]
    ds_target_dt = datetime.fromisoformat(ds_target + "T00:00:00+00:00")
    hh_map = {h.get("from"): h for h in case.get("history_hourly", [])}
    t_dip, t_late = extract_pre_features(hh_map, ds_target_dt, v1)
    t_edt = is_us_edt(ds_target)
    t_v1_cap = clip(math.log(v1), LOG_V1_MIN, LOG_V1_MAX)

    notes_str = (str(case.get("notes", "")) + " " + str(case.get("game", {}).get("name", ""))).lower()
    case_is_rolling = ("rolling" in notes_str) or ("midnight local" in notes_str) or ("12 am local" in notes_str)
    target_console_only = bool(re.search(r"ps5|ps4|xbox|switch", notes_str)) and not bool(
        re.search(r"\b(pc,|pc/|on pc|steam)\b", notes_str)
    )

    valid_launches = []
    for c in case.get("comparables", []):
        unlock_text = str(c.get("unlock", "")).lower()
        plat_text = str(c.get("platforms", "")).lower()
        if "not a launch" in unlock_text or "target-history" in unlock_text:
            continue

        ds = str(c.get("day_start", ""))[:10]
        if not ds:
            continue
        ds_dt = datetime.fromisoformat(ds + "T00:00:00+00:00")
        cmap = {h.get("from"): h for h in c.get("hourly", [])}
        pts = [cmap.get((ds_dt + timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%SZ")) for k in range(24)]
        if any(pt is None for pt in pts):
            continue

        v1_comp = max(1.0, float(pts[0].get("viewers", 1)))
        peak_comp = max(float(pt.get("viewers", 0)) for pt in pts)
        is_rolling = ("rolling" in unlock_text) or ("midnight local" in unlock_text)

        if case_is_rolling:
            if "early-access-contaminated" in unlock_text or (
                "early access" in unlock_text and "no early access" not in unlock_text
            ):
                continue
            if "tier: weak" in unlock_text:
                continue
            is_single = (
                ("single global" in unlock_text)
                or ("single time" in unlock_text)
                or ("simultaneous" in unlock_text)
            )
            if is_single and not is_rolling:
                continue
            if v1_comp < 15000.0 and (peak_comp / v1_comp) > 15.0:
                continue

        c_v1_cap = clip(math.log(v1_comp), LOG_V1_MIN, LOG_V1_MAX)
        dv1 = t_v1_cap - c_v1_cap
        w = math.exp(-0.5 * (dv1 / 1.8) ** 2)

        if target_console_only and ("pc" not in plat_text):
            w *= 3.5
        if is_rolling:
            w *= 2.0
        if ("pc" in plat_text and (
            "simultaneous" in unlock_text or "single global" in unlock_text or "single time" in unlock_text
        )) or ("pc unlock later" in unlock_text):
            w *= 0.25

        c_dip, c_late = extract_pre_features(cmap, ds_dt, v1_comp)
        c_edt = is_us_edt(ds)
        valid_launches.append({
            "v1": v1_comp,
            "pts": pts,
            "w": max(1e-6, w),
            "dip": c_dip,
            "late": c_late,
            "edt": c_edt,
            "v1_cap": c_v1_cap,
        })

    return valid_launches, (t_dip, t_late, t_edt, t_v1_cap)


def block_features(h: int, dip: float, late: float, edt: float, v1_cap: float) -> list[float]:
    """Returns horizon-block feature vector with leading unpenalized intercept."""
    if h <= 4:
        return [1.0, late, dip]
    if h <= 7:
        return [1.0, dip, edt, v1_cap]
    return [1.0, dip, v1_cap]


def estimate_hour_location_scale(
    h: int, valid_launches: list[dict], target_feats: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Fits Weighted Block Ridge + exact Sherman-Morrison LOO residuals + MAD/IQR scale."""
    h_idx = h - 1
    ys = [math.log(max(1.0, float(vl["pts"][h_idx].get("viewers", 1))) / vl["v1"]) for vl in valid_launches]
    raw_ws = [vl["w"] for vl in valid_launches]
    total_w = sum(raw_ws)
    ws = [w / total_w for w in raw_ws]

    if len(valid_launches) < 3:
        med = weighted_quantile(ys, ws, 0.50)
        q25 = weighted_quantile(ys, ws, 0.25)
        q75 = weighted_quantile(ys, ws, 0.75)
        sig = max(0.05, (q75 - q25) / 1.349, 0.04 * math.sqrt(max(1, h - 1)))
        return med, sig

    t_dip, t_late, t_edt, t_v1_cap = target_feats
    xt = block_features(h, t_dip, t_late, t_edt, t_v1_cap)
    X = [block_features(h, vl["dip"], vl["late"], vl["edt"], vl["v1_cap"]) for vl in valid_launches]

    p_dim = len(xt)
    n = len(valid_launches)
    xtwx = [[0.0] * p_dim for _ in range(p_dim)]
    xtwy = [0.0] * p_dim
    for i in range(n):
        wi = ws[i]
        xi = X[i]
        yi = ys[i]
        for a in range(p_dim):
            xtwy[a] += wi * xi[a] * yi
            for b in range(p_dim):
                xtwx[a][b] += wi * xi[a] * xi[b]

    # Variance-scaled Ridge penalty strictly on slopes (intercept j=0 left unpenalized)
    for j in range(1, p_dim):
        mean_j = sum(ws[i] * X[i][j] for i in range(n))
        var_j = sum(ws[i] * (X[i][j] - mean_j) ** 2 for i in range(n))
        xtwx[j][j] += RIDGE_LAMBDA * max(1e-4, var_j)

    a_inv = invert_matrix(xtwx)
    beta = [sum(a_inv[a][b] * xtwy[b] for b in range(p_dim)) for a in range(p_dim)]
    mu_ridge = sum(xt[a] * beta[a] for a in range(p_dim))

    # Exact O(N) Sherman-Morrison LOO residuals
    loo_res = []
    for i in range(n):
        xi = X[i]
        yhat_i = sum(xi[a] * beta[a] for a in range(p_dim))
        a_inv_xi = [sum(a_inv[a][b] * xi[b] for b in range(p_dim)) for a in range(p_dim)]
        h_i = sum(xi[a] * a_inv_xi[a] for a in range(p_dim))
        denom = max(0.25, 1.0 - ws[i] * h_i)
        loo_res.append((ys[i] - yhat_i) / denom)

    med_res = weighted_quantile(loo_res, ws, 0.50)
    med_raw = weighted_quantile(ys, ws, 0.50)
    mu_h = 0.90 * (mu_ridge + med_res) + 0.10 * med_raw

    abs_dev = [abs(r - med_res) for r in loo_res]
    mad = weighted_quantile(abs_dev, ws, 0.50) * 1.4826
    q25 = weighted_quantile(loo_res, ws, 0.25)
    q75 = weighted_quantile(loo_res, ws, 0.75)
    iqr_sig = max(0.04, (q75 - q25) / 1.349)
    sig_h = 0.95 * max(0.05, 0.5 * mad + 0.5 * iqr_sig, 0.03 * math.sqrt(max(1, h - 1)))
    return mu_h, sig_h


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")

    known = case.get("known", [])
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")

    v1 = max(1.0, float(first.get("viewers", 1)))
    valid_launches, target_feats = select_comparables(case, v1)

    mus: dict[int, float] = {}
    raw_sigs: dict[int, float] = {}
    if valid_launches:
        for h in range(2, 25):
            mus[h], raw_sigs[h] = estimate_hour_location_scale(h, valid_launches, target_feats)
    else:
        for h in range(2, 25):
            mus[h] = 0.0
            raw_sigs[h] = 0.15 * math.sqrt(max(1, h - 1))

    # [0.25, 0.50, 0.25] temporal smoothing across hours 2..24
    smooth_sigs: dict[int, float] = {}
    for h in range(2, 25):
        s_prev = raw_sigs[h - 1] if h > 2 else raw_sigs[h]
        s_next = raw_sigs[h + 1] if h < 24 else raw_sigs[h]
        smooth_sigs[h] = 0.25 * s_prev + 0.50 * raw_sigs[h] + 0.25 * s_next

    hours_rows = []
    for row in case.get("forecast", []):
        h = row.get("hour")
        q_vals = [round2(v1 * math.exp(mus[h] + z * smooth_sigs[h])) for z in Z]
        for k in range(len(q_vals)):
            q_vals[k] = clamp_bounds(q_vals[k])
            if k > 0 and q_vals[k] < q_vals[k - 1]:
                q_vals[k] = q_vals[k - 1]
        hours_rows.append({"hour": h, "q": q_vals})

    return {"hours": hours_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_file = OUTPUT_DIR / f"{path.stem}.out"
        out_file.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
