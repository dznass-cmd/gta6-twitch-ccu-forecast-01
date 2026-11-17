"""gta6-twitch-ccu-forecast-01: Algorix Diurnal Kernel Quantile Forecaster.

Predicts Twitch CCU quantiles [0.05, 0.25, 0.50, 0.75, 0.95] for launch day (hours 2..24).
Optimized for minimum pinball loss under Python 3.13 standard library.
"""

import json
import math
import os
from pathlib import Path
import sys

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def weighted_quantile(sorted_vals: list[float], sorted_weights: list[float], total_w: float, p: float) -> float:
    """Computes quantile p from pre-sorted values and matching weights."""
    target = p * total_w
    cum = 0.0
    for v, w in zip(sorted_vals, sorted_weights):
        cum += w
        if cum >= target:
            return v
    return sorted_vals[-1]


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")

    known = case.get("known", [])
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")

    v1 = max(1.0, float(first.get("viewers", 1)))
    ch1 = max(1.0, float(first.get("channels", 1)))

    notes_str = (str(case.get("notes", "")) + " " + str(case.get("game", {}).get("name", ""))).lower()
    case_is_rolling = ("rolling" in notes_str) or ("midnight local" in notes_str)

    valid_launches = []
    for c in case.get("comparables", []):
        ds = str(c.get("day_start", ""))[:10]
        if not ds:
            continue
        hourly = {h.get("from"): h for h in c.get("hourly", []) if str(h.get("from", "")).startswith(ds)}
        pts = [hourly.get(f"{ds}T{h:02d}:00:00Z") for h in range(24)]
        if any(p is None for p in pts):
            continue

        v1_comp = max(1.0, float(pts[0].get("viewers", 1)))
        ch1_comp = max(1.0, float(pts[0].get("channels", 1)))

        unlock_text = str(c.get("unlock", "")).lower()
        is_rolling = ("rolling" in unlock_text) or ("midnight local" in unlock_text)
        is_comp_tier = "tier: comparable" in unlock_text

        dv = math.log(v1) - math.log(v1_comp)
        dch = math.log(ch1) - math.log(ch1_comp)
        w = math.exp(-0.5 * (dv * dv + 0.05 * dch * dch))
        if case_is_rolling and is_rolling:
            w *= math.exp(1.2)
        if is_comp_tier:
            w *= 1.2

        valid_launches.append({
            "v1": v1_comp,
            "pts": pts,
            "weight": max(1e-5, w)
        })

    hours_rows = []
    for row in case.get("forecast", []):
        h = row.get("hour")
        h_idx = h - 1

        if valid_launches and 0 <= h_idx < 24:
            log_ratios = [
                math.log(max(1e-6, vl["pts"][h_idx].get("viewers", 1) / vl["v1"]))
                for vl in valid_launches
            ]
            weights = [vl["weight"] for vl in valid_launches]

            sorted_indices = sorted(range(len(log_ratios)), key=lambda i: log_ratios[i])
            sorted_vals = [log_ratios[i] for i in sorted_indices]
            sorted_w = [weights[i] for i in sorted_indices]
            total_w = sum(sorted_w)

            med = weighted_quantile(sorted_vals, sorted_w, total_w, 0.50)
            q25 = weighted_quantile(sorted_vals, sorted_w, total_w, 0.25)
            q75 = weighted_quantile(sorted_vals, sorted_w, total_w, 0.75)

            rob_sigma = max(0.05, (q75 - q25) / 1.349)
            sigma_eff = max(rob_sigma, 0.04 * math.sqrt(max(1, h - 1)))

            q_vals = [round2(v1 * math.exp(med + z * sigma_eff)) for z in Z]
        else:
            # Fallback to diffusion persistence if comparables missing
            q_vals = [round2(v1 * math.exp(z * 0.15 * math.sqrt(max(1, h - 1)))) for z in Z]

        # Enforce strict non-decreasing bounds within [0.01, 1e9]
        for k in range(len(q_vals)):
            q_vals[k] = min(1e9, max(0.01, q_vals[k]))
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
