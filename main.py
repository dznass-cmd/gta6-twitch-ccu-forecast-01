"""gta6-twitch-ccu-forecast-01: Weighted Block Ridge with Sherman-Morrison LOO and Decoupled Tail Calibration."""

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sys

# Restore /bot on sys.path when the grader launches Python with -P.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
LOG_V1_MIN = math.log(15000.0)
LOG_V1_MAX = math.log(900000.0)
RIDGE_EARLY = 1.0
RIDGE_LATE = 0.5
BW_V1 = 1.8
W_CONSOLE = 2.0
BW_SMOOTH = 1.5
BW_TAIL_POOL = 2.0
TAIL_BASE_Z = 1.96
TAIL_EMP_MIX = 0.75


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def parse_day_start(raw: object) -> datetime:
    """Parses ISO date or timestamp string into a normalized UTC-aware datetime."""
    text = str(raw or "").strip()
    if len(text) == 10:
        return datetime.fromisoformat(f"{text}T00:00:00+00:00")
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def round_half_up_2(val: float) -> float | int:
    scaled = val * 100.0
    floored = math.floor(scaled)
    if scaled - floored >= 0.5:
        floored += 1
    rounded = floored / 100.0
    return int(rounded) if rounded.is_integer() else rounded


def clip_output_bounds(val: float | int) -> float | int:
    if not math.isfinite(val):
        return 0.01
    if val < 0.01:
        return 0.01
    if val > 1e9:
        return 1e9
    return val


def compute_pre_checkpoint_features(
    hourly_lookup: dict[str, dict], anchor_dt: datetime, v1_val: float, ch1_val: float | None = None
) -> tuple[float, float, float]:
    """Computes log viewer ratios (dip at t=-6..-3, late at t=-2..-1) and channel surge ratio."""
    dip_vals = []
    for off in range(-6, -2):
        ts_key = (anchor_dt + timedelta(hours=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = hourly_lookup.get(ts_key)
        if row is not None:
            dip_vals.append(math.log(max(1.0, float(row.get("viewers", 1)))))
    v_dip = math.exp(sum(dip_vals) / len(dip_vals)) if dip_vals else v1_val

    late_vals = []
    late_ch_vals = []
    for off in range(-2, 0):
        ts_key = (anchor_dt + timedelta(hours=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = hourly_lookup.get(ts_key)
        if row is not None:
            late_vals.append(math.log(max(1.0, float(row.get("viewers", 1)))))
            ch_raw = row.get("channels")
            if ch_raw is not None and float(ch_raw) > 0.0:
                late_ch_vals.append(math.log(float(ch_raw)))
    v_late = math.exp(sum(late_vals) / len(late_vals)) if late_vals else v1_val

    if ch1_val is None:
        h1_row = hourly_lookup.get(anchor_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if h1_row is not None:
            ch1_val = h1_row.get("channels")

    dip = clamp(math.log(v_dip / v1_val), -0.8, 1.5)
    late = clamp(math.log(v_late / v1_val), -0.7, 0.8)
    if ch1_val is not None and float(ch1_val) > 0.0 and late_ch_vals:
        mean_late_ch = sum(late_ch_vals) / len(late_ch_vals)
        ch_late = clamp(math.log(float(ch1_val)) - mean_late_ch, -0.45, 0.45)
    else:
        ch_late = 0.0
    return dip, late, ch_late


def weighted_quantile(values: list[float], weights: list[float], prob: float) -> float:
    """Weighted linear-midpoint quantile interpolation."""
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total_w = sum(w for _, w in ordered)
    if total_w <= 0.0:
        return ordered[len(ordered) // 2][0]

    midpoints = []
    acc = 0.0
    for _, w in ordered:
        midpoints.append((acc + 0.5 * w) / total_w)
        acc += w

    if prob <= midpoints[0]:
        return ordered[0][0]
    if prob >= midpoints[-1]:
        return ordered[-1][0]

    for idx in range(len(ordered) - 1):
        lo_m, hi_m = midpoints[idx], midpoints[idx + 1]
        if lo_m <= prob <= hi_m:
            width = hi_m - lo_m
            ratio = 0.0 if width <= 0.0 else (prob - lo_m) / width
            return ordered[idx][0] + ratio * (ordered[idx + 1][0] - ordered[idx][0])
    return ordered[-1][0]


def gauss_jordan_inverse(matrix: list[list[float]]) -> list[list[float]]:
    dim = len(matrix)
    tableau = [
        matrix[r][:] + [1.0 if r == c else 0.0 for c in range(dim)]
        for r in range(dim)
    ]
    for col in range(dim):
        piv_row = max(range(col, dim), key=lambda r: abs(tableau[r][col]))
        if piv_row != col:
            tableau[col], tableau[piv_row] = tableau[piv_row], tableau[col]
        if abs(tableau[col][col]) < 1e-12:
            tableau[col][col] = 1e-12
        inv_diag = 1.0 / tableau[col][col]
        for j in range(col, 2 * dim):
            tableau[col][j] *= inv_diag
        for r in range(dim):
            if r == col:
                continue
            coeff = tableau[r][col]
            if coeff != 0.0:
                for j in range(col, 2 * dim):
                    tableau[r][j] -= coeff * tableau[col][j]
    return [row[dim:] for row in tableau]


def build_block_design_row(
    hour: int, dip: float, late: float, ch_late: float, v1_cap: float
) -> list[float]:
    if hour <= 4:
        return [1.0, late, ch_late]
    return [1.0, dip, v1_cap]


def prepare_comparables(
    case: dict, v1: float, ch1: float | None, target_dt: datetime
) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Filters comparable launches, builds weights, and extracts target features."""
    hh_lookup = {row.get("from"): row for row in case.get("history_hourly", [])}
    t_dip, t_late, t_ch_late = compute_pre_checkpoint_features(hh_lookup, target_dt, v1, ch1)
    t_v1_cap = clamp(math.log(v1), LOG_V1_MIN, LOG_V1_MAX)

    game_name = str(case.get("game", {}).get("name", ""))
    notes_lower = f"{case.get('notes', '')} {game_name}".lower()
    is_target_rolling = (
        ("rolling" in notes_lower)
        or ("midnight local" in notes_lower)
        or ("12 am local" in notes_lower)
    )
    has_console_tag = bool(re.search(r"ps5|ps4|xbox|switch", notes_lower))
    has_pc_tag = bool(re.search(r"\b(?:pc|steam)\b", notes_lower))
    target_console_only = has_console_tag and (
        ("console only" in notes_lower) or not has_pc_tag
    )

    pool: list[dict] = []
    for comp in case.get("comparables", []):
        unlock_str = str(comp.get("unlock", "")).lower()
        plat_str = str(comp.get("platforms", "")).lower()
        if "not a launch" in unlock_str or "target-history" in unlock_str:
            continue

        raw_ds = str(comp.get("day_start", "")).strip()
        if not raw_ds:
            continue
        comp_dt = parse_day_start(raw_ds)
        comp_map = {row.get("from"): row for row in comp.get("hourly", [])}
        day_pts = [
            comp_map.get((comp_dt + timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            for k in range(24)
        ]
        if any(pt is None for pt in day_pts):
            continue

        v1_comp = max(1.0, float(day_pts[0].get("viewers", 1)))
        peak_comp = max(float(pt.get("viewers", 0)) for pt in day_pts)
        comp_rolling = ("rolling" in unlock_str) or ("midnight local" in unlock_str)
        has_campaign_ea = "campaign early access" in unlock_str

        if is_target_rolling:
            if "early-access-contaminated" in unlock_str or (
                "early access" in unlock_str
                and "no early access" not in unlock_str
                and not has_campaign_ea
            ):
                continue
            if "tier: weak" in unlock_str:
                continue
            comp_single = (
                ("single global" in unlock_str)
                or ("single time" in unlock_str)
                or ("simultaneous" in unlock_str)
            )
            if comp_single and not comp_rolling:
                continue
            if v1_comp < 15000.0 and (peak_comp / v1_comp) > 15.0:
                continue

        c_v1_cap = clamp(math.log(v1_comp), LOG_V1_MIN, LOG_V1_MAX)
        weight = math.exp(-0.5 * ((t_v1_cap - c_v1_cap) / BW_V1) ** 2)

        if target_console_only and ("pc" not in plat_str):
            weight *= W_CONSOLE
        if comp_rolling:
            weight *= 2.0
        if has_campaign_ea:
            weight *= 0.60
        if (
            "pc" in plat_str
            and (
                "simultaneous" in unlock_str
                or "single global" in unlock_str
                or "single time" in unlock_str
            )
        ) or ("pc unlock later" in unlock_str):
            weight *= 0.25

        c_dip, c_late, c_ch_late = compute_pre_checkpoint_features(
            comp_map, comp_dt, v1_comp, day_pts[0].get("channels")
        )
        pool.append({
            "v1": v1_comp,
            "pts": day_pts,
            "w": max(1e-6, weight),
            "dip": c_dip,
            "late": c_late,
            "ch_late": c_ch_late,
            "v1_cap": c_v1_cap,
        })

    return pool, (t_dip, t_late, t_ch_late, t_v1_cap)


def fit_hour_ridge_loo(
    hour: int,
    pool: list[dict],
    ws: list[float],
    target_feats: tuple[float, float, float, float],
) -> tuple[float, float, float, list[float]]:
    """Fits Weighted Block Ridge and computes exact Sherman-Morrison LOO residuals, scale, and target leverage."""
    idx = hour - 1
    ys = [math.log(max(1.0, float(item["pts"][idx].get("viewers", 1))) / item["v1"]) for item in pool]

    if len(pool) < 3:
        med = weighted_quantile(ys, ws, 0.50)
        q25 = weighted_quantile(ys, ws, 0.25)
        q75 = weighted_quantile(ys, ws, 0.75)
        sig = max(0.05, (q75 - q25) / 1.349, 0.04 * math.sqrt(max(1, hour - 1)))
        return med, sig, 1.0, [y - med for y in ys]

    t_dip, t_late, t_ch_late, t_v1_cap = target_feats
    x_target = build_block_design_row(hour, t_dip, t_late, t_ch_late, t_v1_cap)
    x_rows = [
        build_block_design_row(hour, item["dip"], item["late"], item["ch_late"], item["v1_cap"])
        for item in pool
    ]

    p_dim = len(x_target)
    n = len(pool)
    xtwx = [[0.0] * p_dim for _ in range(p_dim)]
    xtwy = [0.0] * p_dim
    for i in range(n):
        wi = ws[i]
        xi = x_rows[i]
        yi = ys[i]
        for a in range(p_dim):
            xtwy[a] += wi * xi[a] * yi
            for b in range(p_dim):
                xtwx[a][b] += wi * xi[a] * xi[b]

    ridge_lam = RIDGE_EARLY if hour <= 4 else RIDGE_LATE
    for j in range(1, p_dim):
        mean_j = sum(ws[i] * x_rows[i][j] for i in range(n))
        var_j = sum(ws[i] * (x_rows[i][j] - mean_j) ** 2 for i in range(n))
        xtwx[j][j] += ridge_lam * max(1e-4, var_j)

    a_inv = gauss_jordan_inverse(xtwx)
    beta = [sum(a_inv[a][b] * xtwy[b] for b in range(p_dim)) for a in range(p_dim)]
    mu_ridge = sum(x_target[a] * beta[a] for a in range(p_dim))

    a_inv_xt = [sum(a_inv[a][b] * x_target[b] for b in range(p_dim)) for a in range(p_dim)]
    h_target = max(0.0, sum(x_target[a] * a_inv_xt[a] for a in range(p_dim)))

    loo_res = []
    for i in range(n):
        xi = x_rows[i]
        yhat_i = sum(xi[a] * beta[a] for a in range(p_dim))
        a_inv_xi = [sum(a_inv[a][b] * xi[b] for b in range(p_dim)) for a in range(p_dim)]
        h_i = sum(xi[a] * a_inv_xi[a] for a in range(p_dim))
        denom = max(0.25, 1.0 - ws[i] * h_i)
        loo_res.append((ys[i] - yhat_i) / denom)

    med_res = weighted_quantile(loo_res, ws, 0.50)
    mu_h = mu_ridge + med_res
    centered = [r - med_res for r in loo_res]

    abs_dev = [abs(r) for r in centered]
    mad = weighted_quantile(abs_dev, ws, 0.50) * 1.4826
    q25 = weighted_quantile(centered, ws, 0.25)
    q75 = weighted_quantile(centered, ws, 0.75)
    iqr_sig = max(0.04, (q75 - q25) / 1.349)
    rob_sig = 0.5 * mad + 0.5 * iqr_sig
    rms = math.sqrt(sum(ws[i] * (centered[i] ** 2) for i in range(n)))
    sig_h = max(0.05, 0.75 * rob_sig + 0.25 * rms, 0.030 * math.sqrt(max(1, hour - 1)))
    return mu_h, sig_h, h_target, centered


def apply_intra_hour_momentum(
    case: dict, target_dt: datetime, v1: float, mus: dict[int, float]
) -> None:
    """Adjusts h=2,3 location when minute-level h=1 readings show clear intra-hour drift."""
    h1_end_dt = target_dt + timedelta(hours=1)
    h1_samples = []
    for pt in case.get("history", []):
        at_raw = pt.get("at")
        v_raw = pt.get("viewers")
        if not at_raw or v_raw is None:
            continue
        at_dt = parse_day_start(at_raw)
        if target_dt <= at_dt < h1_end_dt:
            h1_samples.append((at_dt, max(1.0, float(v_raw))))

    if len(h1_samples) >= 20:
        h1_samples.sort(key=lambda pair: pair[0])
        tail_count = max(5, len(h1_samples) // 4)
        v_tail = sum(v for _, v in h1_samples[-tail_count:]) / tail_count
        intra = clamp(math.log(v_tail / v1), -0.35, 0.35)
        mus[2] += 0.35 * intra
        mus[3] += 0.15 * intra


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")

    known = case.get("known", [])
    first = next((row for row in known if row.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")

    v1 = max(1.0, float(first.get("viewers", 1)))
    ch1 = first.get("channels")
    target_dt = parse_day_start(case.get("day_start", ""))

    pool, target_feats = prepare_comparables(case, v1, ch1, target_dt)
    mus: dict[int, float] = {}
    raw_sigs: dict[int, float] = {}
    h_lev: dict[int, float] = {}
    centered_loo: dict[int, list[float]] = {}

    if pool:
        raw_ws = [item["w"] for item in pool]
        w_sum = sum(raw_ws)
        ws = [w / w_sum for w in raw_ws]
        n_eff = 1.0 / sum(wi ** 2 for wi in ws)
        for h in range(2, 25):
            mus[h], raw_sigs[h], h_lev[h], centered_loo[h] = fit_hour_ridge_loo(
                h, pool, ws, target_feats
            )
    else:
        ws = []
        n_eff = 2.0
        for h in range(2, 25):
            mus[h] = 0.0
            raw_sigs[h] = 0.15 * math.sqrt(max(1, h - 1))
            h_lev[h] = 1.0
            centered_loo[h] = []

    apply_intra_hour_momentum(case, target_dt, v1, mus)

    smooth_sigs: dict[int, float] = {}
    for h in range(2, 25):
        w_acc = 0.0
        s_acc = 0.0
        for hp in range(2, 25):
            wh = math.exp(-0.5 * ((h - hp) / BW_SMOOTH) ** 2)
            w_acc += wh
            s_acc += wh * raw_sigs[hp]
        smooth_sigs[h] = s_acc / w_acc

    hours_rows = []
    for row in case.get("forecast", []):
        h = int(row.get("hour"))
        sig_h = smooth_sigs[h]
        if pool:
            pool_vals = []
            pool_ws = []
            for hp in range(2, 25):
                wh = math.exp(-0.5 * ((h - hp) / BW_TAIL_POOL) ** 2)
                if wh < 0.05:
                    continue
                sig_hp = smooth_sigs[hp]
                res_hp = centered_loo[hp]
                for i, wi in enumerate(ws):
                    pool_vals.append(res_hp[i] / sig_hp)
                    pool_ws.append(wh * wi)
            q05_e = weighted_quantile(pool_vals, pool_ws, 0.05)
            q95_e = weighted_quantile(pool_vals, pool_ws, 0.95)
            # Blend empirical standardized residuals with normal tail floor (-/+1.96)
            zlo = min(-TAIL_BASE_Z, TAIL_EMP_MIX * q05_e + (1.0 - TAIL_EMP_MIX) * (-TAIL_BASE_Z))
            zhi = max(+TAIL_BASE_Z, TAIL_EMP_MIX * q95_e + (1.0 - TAIL_EMP_MIX) * (+TAIL_BASE_Z))
            lev_ratio = min(6.0, h_lev[h]) / max(2.0, n_eff)
            tail_mult = math.sqrt(1.0 + 0.75 * lev_ratio) * (1.0 + 0.50 / max(2.0, n_eff))
            offsets = [
                zlo * sig_h * tail_mult,
                -0.48 * sig_h,
                0.0,
                0.48 * sig_h,
                zhi * sig_h * tail_mult,
            ]
        else:
            offsets = [-TAIL_BASE_Z * sig_h, -0.48 * sig_h, 0.0, 0.48 * sig_h, TAIL_BASE_Z * sig_h]

        q_vals = [clip_output_bounds(round_half_up_2(v1 * math.exp(mus[h] + off))) for off in offsets]
        for k in range(1, len(q_vals)):
            if q_vals[k] < q_vals[k - 1]:
                q_vals[k] = q_vals[k - 1]
        hours_rows.append({"hour": h, "q": q_vals})

    return {"hours": hours_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        case_data = json.loads(in_path.read_text(encoding="utf-8"))
        out_data = forecast_case(case_data, in_path.name)
        out_path = OUTPUT_DIR / f"{in_path.stem}.out"
        out_path.write_text(json.dumps(out_data, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
