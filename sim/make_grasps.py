"""Grasp library: thousands of real table pickups (scripted, floppy bag), each
carried to a ready pose.  The throw env starts every episode from one of these,
so the policy learns to throw from whatever grip it actually got.

    python sim/make_grasps.py [n]      -> sim/assets/grasps.npz
"""
import math, multiprocessing as mp, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from skills import Sim, pick
from scene import Bag
from pick_eval import random_pose
import control as C

READY = np.radians([0.0, 95.0, -75.0, -40.0, 0.0])


def one(seed):
    rng = np.random.default_rng(seed)
    s = Sim(Bag(), bag_pose=random_pose(rng))
    ok, info = pick(s, "ridge", press=rng.uniform(0.003, 0.009), pose_noise=rng.uniform(0.0, 0.008), rng=rng)
    if not ok:
        return None
    ready = READY + rng.uniform(-0.15, 0.15, 5) * np.array([1.5, 1, 1, 1, 2])
    s.move_joints(ready, rng.uniform(0.9, 1.3))
    s.settle(0.3)
    pts = s.bag_points()
    R = s.d.xmat[s.m.body("grip").id].reshape(3, 3)
    rel = (R.T @ (pts - s.tcp()).T).T                        # bag points in the gripper frame
    if pts[:, 2].min() < 0.05 or np.linalg.norm(rel[0]) > 0.06 or s.pinched < 0:
        return None                                          # lost it while carrying
    return dict(qpos=s.d.qpos.copy(), qvel=s.d.qvel.copy(), q_tgt=s.q_tgt.copy(), rel=rel.ravel(),
                pinch=np.array([s.pinched]), pinch_data=s.m.eq_data[s.pinch_eq[s.pinched]].copy(),
                phi_pinch=np.array([s.phi_pinch]))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t = time.time()
    out = []
    with mp.Pool(os.cpu_count()) as pool:
        for r in pool.imap_unordered(one, range(int(n * 1.15))):
            if r is not None:
                out.append(r)
            if len(out) >= n:
                break
    keys = out[0].keys()
    np.savez_compressed(os.path.join(HERE, "assets", "grasps.npz"), **{k: np.array([o[k] for o in out]) for k in keys})
    rel = np.array([o["rel"] for o in out]).reshape(len(out), -1, 3)
    print(f"{len(out)} grasps in {time.time() - t:.0f} s; bag centre in the gripper frame: "
          f"mean {np.round(rel[:, 0].mean(0) * 1000, 1)} mm, spread {np.round(rel[:, 0].std(0) * 1000, 1)} mm")


if __name__ == "__main__":
    main()
