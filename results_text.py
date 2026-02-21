#!/usr/bin/env python3
"""
Extract a compact, shareable TEXT report from training results JSON
(Updated for NEW environment schema: TTC-based, reason ∈ {goal, collision, timeout})
"""

import argparse
import json
import math
import os
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def is_num(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def f(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def i(x, default=None) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return default


def pct(a, b) -> str:
    if b <= 0:
        return "n/a"
    return f"{100.0 * a / b:.2f}%"


def fmt_stat(arr: np.ndarray, name: str, unit: str = "") -> str:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name}: n=0"
    q = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return (
        f"{name}: n={arr.size} | mean={arr.mean():.4g}{unit} std={arr.std(ddof=0):.4g}{unit} "
        f"| min={q[0]:.4g}{unit} p10={q[1]:.4g}{unit} p25={q[2]:.4g}{unit} "
        f"med={q[3]:.4g}{unit} p75={q[4]:.4g}{unit} p90={q[5]:.4g}{unit} max={q[6]:.4g}{unit}"
    )


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for k in range(len(x)):
        a = max(0, k - w + 1)
        seg = x[a : k + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[k] = seg.mean()
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    denom = np.sqrt((aa**2).mean()) * np.sqrt((bb**2).mean())
    return float((aa * bb).mean() / denom) if denom > 0 else np.nan


# -----------------------------
# Main extraction
# -----------------------------
def extract_report(episodes: List[Dict[str, Any]], every: int = 50, topk: int = 20) -> str:
    episodes = sorted(episodes, key=lambda e: f(e.get("episode"), np.nan))
    n = len(episodes)

    ep_nums = np.array([f(e.get("episode")) for e in episodes])
    rewards = np.array([f(e.get("reward")) for e in episodes])
    lengths = np.array([f(e.get("length")) for e in episodes])
    times = np.array([f(e.get("time")) for e in episodes])

    actor_loss = np.array([f(e.get("actor_loss")) for e in episodes])
    critic_loss = np.array([f(e.get("critic_loss")) for e in episodes])
    ent_coef = np.array([f(e.get("ent_coef")) for e in episodes])

    min_obs = np.array([f(e.get("min_obstacle_depth")) for e in episodes])
    min_ultra = np.array([f(e.get("min_ultrasonic_distance")) for e in episodes])

    ttc = np.array([f(e.get("ttc")) for e in episodes])
    v_cmd = np.array([f(e.get("v_cmd")) for e in episodes])
    w_cmd = np.array([f(e.get("w_cmd")) for e in episodes])

    reasons = [str(e.get("reason", "unknown")).lower() for e in episodes]
    reason_counts = Counter(reasons)

    # Outcome summary (exact match)
    outcome = Counter()
    for r in reasons:
        if r == "goal":
            outcome["success"] += 1
        elif r == "collision":
            outcome["collision"] += 1
        elif r == "timeout":
            outcome["timeout"] += 1
        else:
            outcome["other"] += 1

    roll100 = rolling_mean(rewards, 100)

    # Top / worst episodes
    idx = np.where(np.isfinite(rewards))[0]
    order = idx[np.argsort(rewards[idx])]
    worst = order[:topk]
    best = order[-topk:][::-1]

    def ep_line(e):
        return (
            f"ep={i(e.get('episode'))} | R={f(e.get('reward')):.3f} | len={f(e.get('length')):.0f} "
            f"| time={f(e.get('time')):.2f}s | reason={e.get('reason')} "
            f"| ttc={f(e.get('ttc')):.3f} | v={f(e.get('v_cmd')):.3f} | w={f(e.get('w_cmd')):.3f} "
            f"| min_obs={f(e.get('min_obstacle_depth')):.3f} | min_ultra={f(e.get('min_ultrasonic_distance')):.3f}"
        )

    # Assemble report
    L = []
    L.append("=== TRAINING RESULTS REPORT (shareable text) ===\n")
    L.append(f"Episodes: {n}")
    L.append(f"Episode range: {int(np.nanmin(ep_nums))} .. {int(np.nanmax(ep_nums))}\n")

    L.append("== Outcome summary ==")
    for k in ["success", "collision", "timeout", "other"]:
        L.append(f"{k:10s}: {outcome.get(k,0):6d} ({pct(outcome.get(k,0), n)})")
    L.append("")

    L.append("== Core stats ==")
    L.append(fmt_stat(rewards, "reward"))
    L.append(fmt_stat(lengths, "episode_length", " steps"))
    L.append(fmt_stat(times, "episode_time", " s"))
    L.append(fmt_stat(actor_loss, "actor_loss"))
    L.append(fmt_stat(critic_loss, "critic_loss"))
    L.append(fmt_stat(ent_coef, "entropy_coef"))
    L.append(fmt_stat(min_ultra, "min_ultrasonic_distance", " m"))
    L.append(fmt_stat(min_obs, "min_obstacle_depth", " m"))
    L.append(fmt_stat(ttc, "ttc", " s"))
    L.append(fmt_stat(v_cmd, "v_cmd"))
    L.append(fmt_stat(w_cmd, "w_cmd"))
    L.append("")

    L.append("== Rolling reward (mean) ==")
    L.append(
        f"rolling100: last={roll100[np.isfinite(roll100)][-1]:.3f} "
        f"| best={np.nanmax(roll100):.3f} | worst={np.nanmin(roll100):.3f}\n"
    )

    L.append("== Quick correlations ==")
    L.append(f"corr(reward, ttc)        = {corr(rewards, ttc):.4f}")
    L.append(f"corr(reward, v_cmd)      = {corr(rewards, v_cmd):.4f}")
    L.append(f"corr(reward, w_cmd)      = {corr(rewards, w_cmd):.4f}")
    L.append(f"corr(reward, min_obs)    = {corr(rewards, min_obs):.4f}")
    L.append(f"corr(reward, min_ultra)  = {corr(rewards, min_ultra):.4f}\n")

    L.append(f"== Top {len(best)} episodes by reward ==")
    for j in best:
        L.append("  " + ep_line(episodes[j]))
    L.append("")

    L.append(f"== Worst {len(worst)} episodes by reward ==")
    for j in worst:
        L.append("  " + ep_line(episodes[j]))
    L.append("")

    L.append(f"== Downsampled per-episode series (every {every}) ==")
    for k in range(0, n, every):
        L.append(ep_line(episodes[k]))

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="training_report.txt")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("JSON must be a non-empty list")

    report = extract_report(data, every=max(1, args.every), topk=max(1, args.topk))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[ok] Report written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
