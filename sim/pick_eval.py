"""Success rate of the scripted table pickup over random bag poses.
    python sim/pick_eval.py [n]"""
import math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from skills import Sim, pick
from scene import Bag


def random_pose(rng):
    r = rng.uniform(0.17, 0.32)
    a = rng.uniform(-1.0, 1.0)
    return (r * math.cos(a), r * math.sin(a), rng.uniform(-180, 180))


def evaluate(strategy, n=100, seed=0, noise=0.005, **kw):
    rng = np.random.default_rng(seed)
    res = {"ok": 0, "unreachable": 0, "dropped": 0}
    for i in range(n):
        s = Sim(Bag(), bag_pose=random_pose(rng))
        ok, info = pick(s, strategy, pose_noise=noise, rng=rng, **kw)
        res["ok" if ok else info["reason"]] += 1
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for strat, kw in (("ridge", {}), ("ridge", {"press": 0.003}), ("ridge", {"press": 0.009})):
        t = time.time()
        r = evaluate(strat, n, **kw)
        print(f"{strat:9s} {str(kw):18s} success {r['ok'] / n:5.0%}   dropped {r['dropped']}  unreachable {r['unreachable']}   ({time.time() - t:.0f} s)", flush=True)
