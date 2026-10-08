"""Compare every run in a group:  python sim/leaderboard.py runs/throw"""
import csv, json, os, sys

group = sys.argv[1] if len(sys.argv) > 1 else "runs"
KEYS = ("points", "in_hole", "success", "dist", "episode_reward")
rows = []
for tag in sorted(os.listdir(group)):
    p = os.path.join(group, tag, "metrics.csv")
    if not os.path.exists(p):
        continue
    data = list(csv.DictReader(open(p)))
    if not data:
        continue
    cfg = json.load(open(os.path.join(group, tag, "config.json")))
    last = data[-1]
    pick = lambda k: next((c for c in last if c.endswith(k)), None)
    score_key = next((pick(k) for k in KEYS if pick(k)), None)
    sign = -1 if (score_key or "").endswith("dist") else 1           # distance: lower is better
    best = max(data, key=lambda r: sign * float(r.get(score_key) or "nan")) if score_key else last
    rows.append((tag, cfg["algo"], int(float(last["step"])), float(last["wall_s"]) / 3600, score_key,
                 float(best.get(score_key, "nan")), float(last.get(score_key, "nan")),
                 {k: float(last[pick(k)]) for k in KEYS if pick(k)}))
lower_better = lambda k: bool(k) and k.endswith("dist")
rows.sort(key=lambda r: r[5] if lower_better(r[4]) else -r[5])
print(f"{'run':18s} {'algo':9s} {'steps':>12s} {'hours':>6s}   {'best':>8s} {'last':>8s}  metric / latest metrics")
for tag, algo, step, h, k, best, last, m in rows:
    print(f"{tag:18s} {algo:9s} {step:12,d} {h:6.2f}   {best:8.3f} {last:8.3f}  {k.split('/')[-1] if k else ''}  "
          + "  ".join(f"{kk} {v:.3f}" for kk, v in m.items()))
