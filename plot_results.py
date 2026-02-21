"""
Training Analysis Plotter (single file)

Reads your JSON list of episode dicts and generates the full analysis suite:
1) Reward vs episode (+ rolling mean) + colored by reason
2) Rolling success rate
3) Episode length vs episode (+ time limit highlighting)
4) Time per episode vs episode
5) Reason distribution (overall bar) + rolling stacked fractions
6) min_dist vs reward scatter (colored by reason)
7) Safety margins: min_obstacle_depth & min_ultrasonic_distance by reason (boxplots)
8) Actor/Critic loss vs episode (lines)
9) ent_coef vs episode
10) Reward vs critic_loss scatter
11) Start-goal distance vs outcome (scatter)
12) Start/Goal position scatter maps colored by reason
13) Obstacle world slots overlay (robot_start/final, goal_start/final, obstacles) for selected episodes
14) Trajectory "story plot" for episodes that have robot_path/goal_path (+ optional gt_obstacles_path)

Usage:
  python plot_training_results.py --json training_results.json --out plots_out --rolling 20 --topk 12

Notes:
- This script assumes your JSON is a list of dicts like the snippet you pasted.
- It is robust to missing fields (it will skip plots/episodes that lack needed keys).
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    n = np.cumsum(np.isfinite(x).astype(float))
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        s = c[i] - (c[j0 - 1] if j0 > 0 else 0.0)
        k = n[i] - (n[j0 - 1] if j0 > 0 else 0.0)
        out[i] = (s / k) if k > 0 else np.nan
    return out


def rolling_fraction(labels, w):
    """
    labels: list[str]
    returns: dict[label] -> np.array fractions per index
    """
    labels = list(labels)
    uniq = sorted(set([l for l in labels if l is not None]))
    n = len(labels)
    out = {u: np.full(n, np.nan, dtype=float) for u in uniq}
    for i in range(n):
        j0 = max(0, i - w + 1)
        window = labels[j0 : i + 1]
        total = len(window)
        if total == 0:
            continue
        cnt = Counter(window)
        for u in uniq:
            out[u][i] = cnt.get(u, 0) / total
    return out


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def reason_palette(reasons):
    """
    Deterministic palette mapping for reasons (no custom colors specified by user),
    so we just use matplotlib default cycle but map consistently.
    """
    uniq = sorted(set(reasons))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    mapping = {}
    for i, r in enumerate(uniq):
        mapping[r] = colors[i % len(colors)] if colors else None
    return mapping


def vec2(p):
    if p is None or len(p) < 2:
        return (np.nan, np.nan)
    return (safe_float(p[0]), safe_float(p[1]))


def dist2(a, b):
    ax, ay = vec2(a)
    bx, by = vec2(b)
    if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(bx) and np.isfinite(by)):
        return np.nan
    return math.hypot(ax - bx, ay - by)


# -----------------------------
# Plot functions
# -----------------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_reward(episodes, outdir, w):
    ep = [e["episode"] for e in episodes]
    rew = [safe_float(e.get("reward")) for e in episodes]
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]
    cmap = reason_palette(reasons)

    plt.figure()
    for r in sorted(set(reasons)):
        idx = [i for i, rr in enumerate(reasons) if rr == r]
        plt.scatter(np.array(ep)[idx], np.array(rew)[idx], label=r, s=22, alpha=0.8)

    rm = rolling_mean(rew, w)
    plt.plot(ep, rm, linewidth=2, label=f"Rolling mean (w={w})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode (colored by termination reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "01_reward_vs_episode.png"))


def plot_success_rate(episodes, outdir, w):
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]
    success = np.array([1.0 if r.lower() == "reached goal" else 0.0 for r in reasons], dtype=float)
    sr = rolling_mean(success, w) * 100.0

    plt.figure()
    plt.plot(ep, sr, linewidth=2)
    plt.ylim(-5, 105)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling success rate (%) (w={w})")
    plt.title("Rolling Success Rate (Reached goal)")
    savefig(os.path.join(outdir, "02_rolling_success_rate.png"))


def plot_length_and_time(episodes, outdir):
    ep = [e["episode"] for e in episodes]
    length = [safe_float(e.get("length")) for e in episodes]
    t = [safe_float(e.get("time")) for e in episodes]
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]

    plt.figure()
    plt.scatter(ep, length, s=22, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Episode length (steps)")
    plt.title("Episode Length vs Episode")
    savefig(os.path.join(outdir, "03_length_vs_episode.png"))

    plt.figure()
    plt.scatter(ep, t, s=22, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Episode wall-time (s)")
    plt.title("Episode Time vs Episode")
    savefig(os.path.join(outdir, "04_time_vs_episode.png"))

    # Optional: show length colored by reason for interpretation
    plt.figure()
    cmap = reason_palette(reasons)
    for r in sorted(set(reasons)):
        idx = [i for i, rr in enumerate(reasons) if rr == r]
        plt.scatter(np.array(ep)[idx], np.array(length)[idx], label=r, s=22, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Episode length (steps)")
    plt.title("Episode Length vs Episode (colored by reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "05_length_vs_episode_by_reason.png"))


def plot_reason_distribution(episodes, outdir, w):
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]
    cnt = Counter(reasons)

    # Overall bar
    plt.figure()
    xs = list(cnt.keys())
    ys = [cnt[k] for k in xs]
    plt.bar(xs, ys)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Termination reason")
    plt.ylabel("Count")
    plt.title("Termination Reasons (overall)")
    savefig(os.path.join(outdir, "06_reason_overall_bar.png"))

    # Rolling stacked fractions
    fr = rolling_fraction(reasons, w)
    plt.figure()
    # stack in sorted reason order for determinism
    order = sorted(fr.keys())
    base = np.zeros(len(ep), dtype=float)
    for r in order:
        y = np.nan_to_num(fr[r], nan=0.0)
        plt.fill_between(ep, base, base + y, step=None, alpha=0.8, label=r)
        base = base + y
    plt.ylim(0, 1.0)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling fraction (w={w})")
    plt.title("Termination Reason Mix Over Training (rolling fractions)")
    plt.legend(loc="upper right", fontsize=8)
    savefig(os.path.join(outdir, "07_reason_rolling_stacked.png"))


def plot_min_dist_vs_reward(episodes, outdir):
    ep = [e["episode"] for e in episodes]
    rew = np.array([safe_float(e.get("reward")) for e in episodes], dtype=float)
    md = np.array([safe_float(e.get("min_dist")) for e in episodes], dtype=float)
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]

    plt.figure()
    for r in sorted(set(reasons)):
        idx = [i for i, rr in enumerate(reasons) if rr == r]
        plt.scatter(md[idx], rew[idx], label=r, s=22, alpha=0.8)
    plt.xlabel("min_dist to goal at termination")
    plt.ylabel("Reward")
    plt.title("Reward vs min_dist (colored by reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "08_reward_vs_min_dist.png"))


def plot_safety_margins_by_reason(episodes, outdir):
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]
    uniq = sorted(set(reasons))

    min_depth = {r: [] for r in uniq}
    min_ultra = {r: [] for r in uniq}
    for e, r in zip(episodes, reasons):
        d = safe_float(e.get("min_obstacle_depth"))
        u = safe_float(e.get("min_ultrasonic_distance"))
        if np.isfinite(d):
            min_depth[r].append(d)
        if np.isfinite(u):
            min_ultra[r].append(u)

    # min_obstacle_depth
    if any(len(v) > 0 for v in min_depth.values()):
        plt.figure()
        data = [min_depth[r] for r in uniq]
        plt.boxplot(data, labels=uniq, showfliers=True)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("min_obstacle_depth")
        plt.title("min_obstacle_depth by termination reason")
        savefig(os.path.join(outdir, "09_min_obstacle_depth_by_reason.png"))

    # min_ultrasonic_distance
    if any(len(v) > 0 for v in min_ultra.values()):
        plt.figure()
        data = [min_ultra[r] for r in uniq]
        plt.boxplot(data, labels=uniq, showfliers=True)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("min_ultrasonic_distance")
        plt.title("min_ultrasonic_distance by termination reason")
        savefig(os.path.join(outdir, "10_min_ultrasonic_distance_by_reason.png"))


def plot_losses_and_entropy(episodes, outdir):
    ep = [e["episode"] for e in episodes]
    actor = [safe_float(e.get("actor_loss")) for e in episodes]
    critic = [safe_float(e.get("critic_loss")) for e in episodes]
    ent = [safe_float(e.get("ent_coef")) for e in episodes]

    # Actor/Critic loss lines
    plt.figure()
    plt.plot(ep, actor, linewidth=2, label="actor_loss")
    plt.plot(ep, critic, linewidth=2, label="critic_loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor/Critic Loss vs Episode")
    plt.legend(loc="best")
    savefig(os.path.join(outdir, "11_losses_vs_episode.png"))

    # ent_coef
    plt.figure()
    plt.plot(ep, ent, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("ent_coef")
    plt.title("Entropy Coefficient vs Episode")
    savefig(os.path.join(outdir, "12_ent_coef_vs_episode.png"))

    # reward vs critic_loss scatter
    rew = np.array([safe_float(e.get("reward")) for e in episodes], dtype=float)
    critic_arr = np.array(critic, dtype=float)
    plt.figure()
    plt.scatter(critic_arr, rew, s=22, alpha=0.8)
    plt.xlabel("critic_loss")
    plt.ylabel("Reward")
    plt.title("Reward vs critic_loss")
    savefig(os.path.join(outdir, "13_reward_vs_critic_loss.png"))


def plot_start_goal_difficulty(episodes, outdir):
    ep = [e["episode"] for e in episodes]
    d0 = []
    rew = []
    reasons = []
    for e in episodes:
        d0.append(dist2(e.get("robot_start"), e.get("goal_start")))
        rew.append(safe_float(e.get("reward")))
        reasons.append(str(e.get("reason", "Unknown")))
    d0 = np.array(d0, dtype=float)
    rew = np.array(rew, dtype=float)

    plt.figure()
    for r in sorted(set(reasons)):
        idx = [i for i, rr in enumerate(reasons) if rr == r]
        plt.scatter(d0[idx], rew[idx], label=r, s=22, alpha=0.8)
    plt.xlabel("Start-goal distance ||goal_start - robot_start||")
    plt.ylabel("Reward")
    plt.title("Difficulty vs Reward (colored by reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "14_start_goal_distance_vs_reward.png"))


def plot_start_goal_maps(episodes, outdir):
    reasons = [str(e.get("reason", "Unknown")) for e in episodes]
    uniq = sorted(set(reasons))

    # robot_start map
    plt.figure()
    for r in uniq:
        xs, ys = [], []
        for e, rr in zip(episodes, reasons):
            if rr != r:
                continue
            x, y = vec2(e.get("robot_start"))
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y)
        if xs:
            plt.scatter(xs, ys, s=22, alpha=0.8, label=r)
    plt.xlabel("robot_start.x")
    plt.ylabel("robot_start.y")
    plt.title("Robot Start Positions (colored by reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "15_robot_start_map.png"))

    # goal_start map
    plt.figure()
    for r in uniq:
        xs, ys = [], []
        for e, rr in zip(episodes, reasons):
            if rr != r:
                continue
            x, y = vec2(e.get("goal_start"))
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y)
        if xs:
            plt.scatter(xs, ys, s=22, alpha=0.8, label=r)
    plt.xlabel("goal_start.x")
    plt.ylabel("goal_start.y")
    plt.title("Goal Start Positions (colored by reason)")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(outdir, "16_goal_start_map.png"))


def pick_representative_episodes(episodes, topk=12):
    """
    Select a small set of representative episodes:
    - best reward successes
    - worst reward collisions
    - a couple timeouts
    - a couple lost marker timeouts
    """
    if not episodes:
        return []

    # categorize
    by_reason = defaultdict(list)
    for e in episodes:
        by_reason[str(e.get("reason", "Unknown"))].append(e)

    def sort_by_reward(es, reverse=True):
        return sorted(es, key=lambda x: safe_float(x.get("reward"), -np.inf), reverse=reverse)

    picks = []

    # successes
    for r in by_reason.keys():
        if r.lower() == "reached goal":
            picks += sort_by_reward(by_reason[r], reverse=True)[: max(1, topk // 6)]

    # worst collisions
    for key in ["Collision with obstacle", "Ultrasonic collision"]:
        if key in by_reason:
            picks += sort_by_reward(by_reason[key], reverse=False)[: max(1, topk // 6)]

    # time limit
    if "Time limit reached" in by_reason:
        picks += sort_by_reward(by_reason["Time limit reached"], reverse=True)[: max(1, topk // 6)]

    # lost marker
    if "Lost marker timeout" in by_reason:
        picks += sort_by_reward(by_reason["Lost marker timeout"], reverse=False)[: max(1, topk // 6)]

    # fill remainder with diverse rewards
    if len(picks) < topk:
        remaining = [e for e in episodes if e not in picks]
        # sample across reward quantiles
        remaining = sorted(remaining, key=lambda x: safe_float(x.get("reward"), 0.0))
        if remaining:
            idxs = np.linspace(0, len(remaining) - 1, num=min(topk - len(picks), len(remaining)), dtype=int)
            picks += [remaining[i] for i in idxs]

    # unique by episode id
    seen = set()
    uniq = []
    for e in picks:
        eid = e.get("episode")
        if eid in seen:
            continue
        seen.add(eid)
        uniq.append(e)
        if len(uniq) >= topk:
            break
    return uniq


def plot_obstacle_world_overlay(episodes, outdir, topk=12):
    reps = pick_representative_episodes(episodes, topk=topk)
    if not reps:
        return

    for e in reps:
        ep = e.get("episode")
        reason = str(e.get("reason", "Unknown"))

        rsx, rsy = vec2(e.get("robot_start"))
        rfx, rfy = vec2(e.get("robot_final"))
        gsx, gsy = vec2(e.get("goal_start"))
        gfx, gfy = vec2(e.get("goal_final"))

        plt.figure()
        # starts/finals
        if np.isfinite(rsx) and np.isfinite(rsy):
            plt.scatter([rsx], [rsy], marker="o", s=60, label="robot_start")
        if np.isfinite(rfx) and np.isfinite(rfy):
            plt.scatter([rfx], [rfy], marker="x", s=60, label="robot_final")
        if np.isfinite(gsx) and np.isfinite(gsy):
            plt.scatter([gsx], [gsy], marker="^", s=60, label="goal_start")
        if np.isfinite(gfx) and np.isfinite(gfy):
            plt.scatter([gfx], [gfy], marker="s", s=60, label="goal_final")

        # obstacles world slots
        slots = e.get("obstacles_world_slots", [])
        ox, oy = [], []
        for s in slots:
            if not isinstance(s, dict):
                continue
            if not s.get("valid", True):
                continue
            wx = safe_float(s.get("wx"))
            wy = safe_float(s.get("wy"))
            if np.isfinite(wx) and np.isfinite(wy):
                ox.append(wx); oy.append(wy)
        if ox:
            plt.scatter(ox, oy, s=60, label="obstacles_world_slots")

        plt.xlabel("x (world)")
        plt.ylabel("y (world)")
        plt.axis("equal")
        plt.title(f"Episode {ep}: world overlay ({reason})")
        plt.legend(loc="best", fontsize=8)
        savefig(os.path.join(outdir, f"17_world_overlay_ep_{ep:05d}.png"))


def plot_trajectory_story(episodes, outdir, topk=12):
    """
    For episodes that contain robot_path, plot:
      - robot_path polyline
      - goal_path polyline (if present)
      - final obstacle world slots (if present)
      - gt_obstacles_path (if present): plot last known positions (or all as faint points)
    """
    candidates = [e for e in episodes if isinstance(e.get("robot_path"), list) and len(e.get("robot_path")) > 1]
    if not candidates:
        return

    # pick representative among those
    reps = pick_representative_episodes(candidates, topk=topk)

    for e in reps:
        ep = e.get("episode")
        reason = str(e.get("reason", "Unknown"))

        rp = e.get("robot_path", [])
        rxy = np.array([[safe_float(p[0]), safe_float(p[1])] for p in rp if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
        if rxy.shape[0] < 2:
            continue

        plt.figure()
        plt.plot(rxy[:, 0], rxy[:, 1], linewidth=2, label="robot_path")

        gp = e.get("goal_path", None)
        if isinstance(gp, list) and len(gp) > 1:
            gxy = np.array([[safe_float(p[0]), safe_float(p[1])] for p in gp if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            if gxy.shape[0] >= 2:
                plt.plot(gxy[:, 0], gxy[:, 1], linewidth=2, label="goal_path")

        # obstacles world slots (final)
        slots = e.get("obstacles_world_slots", [])
        ox, oy = [], []
        for s in slots:
            if isinstance(s, dict) and s.get("valid", True):
                wx = safe_float(s.get("wx"))
                wy = safe_float(s.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy):
                    ox.append(wx); oy.append(wy)
        if ox:
            plt.scatter(ox, oy, s=60, label="obstacles_world_slots")

        # ground-truth obstacles path (if present)
        gt = e.get("gt_obstacles_path", None)
        if isinstance(gt, list) and len(gt) > 0:
            gtx, gty = [], []
            for frame in gt:
                if not isinstance(frame, dict):
                    continue
                for k, v in frame.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        x, y = safe_float(v[0]), safe_float(v[1])
                        if np.isfinite(x) and np.isfinite(y):
                            gtx.append(x); gty.append(y)
            if gtx:
                plt.scatter(gtx, gty, s=12, alpha=0.35, label="gt_obstacles_path (points)")

        plt.xlabel("x (world)")
        plt.ylabel("y (world)")
        plt.axis("equal")
        plt.title(f"Episode {ep}: Trajectory story ({reason})")
        plt.legend(loc="best", fontsize=8)
        savefig(os.path.join(outdir, f"18_trajectory_story_ep_{ep:05d}.png"))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to JSON file containing list of episode dicts.")
    parser.add_argument("--out", default="plots_out", help="Output directory for figures.")
    parser.add_argument("--rolling", type=int, default=20, help="Rolling window size (episodes).")
    parser.add_argument("--topk", type=int, default=12, help="How many representative episodes to plot in overlays/stories.")
    args = parser.parse_args()

    ensure_dir(args.out)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of episode dicts.")

    # Ensure episodes are sorted by episode id (if present)
    data = sorted(data, key=lambda e: safe_float(e.get("episode"), np.inf))

    # Basic plots
    plot_reward(data, args.out, args.rolling)
    plot_success_rate(data, args.out, args.rolling)
    plot_length_and_time(data, args.out)
    plot_reason_distribution(data, args.out, args.rolling)
    plot_min_dist_vs_reward(data, args.out)
    plot_safety_margins_by_reason(data, args.out)
    plot_losses_and_entropy(data, args.out)
    plot_start_goal_difficulty(data, args.out)
    plot_start_goal_maps(data, args.out)

    # World/trajectory diagnostics
    plot_obstacle_world_overlay(data, args.out, topk=args.topk)
    plot_trajectory_story(data, args.out, topk=args.topk)

    print(f"Done. Saved plots to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
