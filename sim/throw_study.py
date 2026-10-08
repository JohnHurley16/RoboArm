"""How far can Capstan Arm 5 throw a beanbag?

For each bag, optimise an open-loop overhand throw in the plane of J2-J4
with the cross-entropy method on CPU MuJoCo.  The real limits are all in the
loop: motor torque-speed curves at 24 V, the PD loop, the capstan ratios, the
servo's grip force (it can only squeeze so hard, so a heavy bag slips), and
the servo's release delay.

    python sim/throw_study.py            -> sim/out/throw_study.json, .md, plots

The throw: each of J2, J3, J4 moves from a wind-up angle to a follow-through
angle along a smooth (minimum-jerk) profile with its own start time and
duration; the gripper opens at the release time.  13 numbers per throw.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C                                  # noqa: E402
from scene import scene_xml, Bag                    # noqa: E402

OUT = os.path.join(HERE, "out")
DT = 0.002
SETTLE = 0.80          # s: wind up from the pickup pose (or hold) before the throw starts
T_THROW = 1.20         # s of commanded motion
T_MAX = 4.0            # s total simulated
# parameter vector: q0 (J2,J3,J4) deg, q1 (J2,J3,J4) deg, t_start (3) s, duration (3) s, t_release s
LO = np.array([60, -110, -90, -10, -110, -90, 0.00, 0.00, 0.00, 0.08, 0.06, 0.04, 0.15])
HI = np.array([120, 110, 70, 120, 110, 70, 0.50, 0.60, 0.70, 0.70, 0.60, 0.50, 1.10])

BAGS = {   # name: Bag
    "mini 2 in, 25 g": Bag(size=0.05, thick=0.016, mass=0.025),
    "medium 3 in, 60 g": Bag(size=0.075, thick=0.02, mass=0.060),
    "4 in, 150 g": Bag(size=0.10, thick=0.022, mass=0.150),
    "regulation 6 in, 450 g": Bag(size=0.152, thick=0.025, mass=0.450),
}

_model = {}


def joint_ids(m):
    return dict(qadr=[m.jnt_qposadr[m.joint(j).id] for j in C.JOINTS],
                dadr=[m.jnt_dofadr[m.joint(j).id] for j in C.JOINTS],
                gq=m.jnt_qposadr[m.joint("G").id],
                bag_q=m.jnt_qposadr[m.joint("bag").id], bag_d=m.jnt_dofadr[m.joint("bag").id],
                act=[m.actuator(j).id for j in C.JOINTS], gact=m.actuator("G").id,
                coupled=[(m.jnt_qposadr[m.joint(k).id], s) for k, s in
                         (("Gl", -1), ("Gku", -1), ("Gkl", 1), ("Gfu", 1), ("Gfl", -1))])


def get_model(bag):
    key = (bag.size, bag.thick, bag.mass)
    if key not in _model:
        m = mujoco.MjModel.from_xml_string(scene_xml(bag=bag, board=None))
        _model[key] = (m, joint_ids(m))
    return _model[key]


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s), (30 * s * s - 60 * s ** 3 + 30 * s ** 4)


def place_grip(m, d, ids, q_deg, bag, phi):
    """Arm at q, gripper at phi, bag held between the pads (pinched near its edge)."""
    d.qpos[:] = 0
    for i, j in enumerate(C.JOINTS):
        d.qpos[ids["qadr"][i]] = math.radians(q_deg[j])
    d.qpos[ids["gq"]] = phi
    for adr, s in ids["coupled"]:
        d.qpos[adr] = s * phi
    mujoco.mj_forward(m, d)
    pu, pl = d.site("pad_u").xpos.copy(), d.site("pad_l").xpos.copy()
    R = d.body("grip").xmat.reshape(3, 3)
    c = (pu + pl) / 2 + R[:, 0] * max(0.0, bag.size / 2 - 0.024)      # bite 24 mm of the bag's edge
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())                                  # bag z along the jaw axis
    d.qpos[ids["bag_q"]:ids["bag_q"] + 3] = c
    d.qpos[ids["bag_q"] + 3:ids["bag_q"] + 7] = quat
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def rollout(params, bag, record=False, grasp=None, record_qpos=False, stall=None):
    """One throw on the shared controller (sim/skills.py): PD + gravity feedforward,
    servo delay, and the pinch latch.  grasp: a grasp-library state (a real
    table pickup); without one the arm first picks the bag off the table."""
    from skills import Sim, pick, throw
    if grasp is not None:
        sim = Sim.from_grasp(grasp, bag=bag)
    else:
        sim = Sim(bag, bag_pose=(0.22, 0.0, 0.0))
        ok, _ = pick(sim)
        if not ok:
            return dict(held=False, distance=float("nan"), slipped=False, released_at=None, landed_xy=[0, 0])
    if stall is not None:
        sim.m.actuator_forcerange[sim.gact] = [-stall, stall]
        C.SERVO["stall_nm"] = stall
    r = throw(sim, params, stop_at_landing=not record)
    r["distance"] = float(r["landed_xy"][0]) if r["held"] else float("nan")
    return r


def score(params, bag, grasp=None):
    r = rollout(params, bag, grasp=grasp)
    if not r["held"] or r["released_at"] is None:
        return -10.0
    s = r["distance"]
    if r["slipped"]:
        s -= 0.3
    return s


def _score_job(args):
    return score(*args)


def cem(bag, iters=28, pop=64, elite=8, seed=0, pool=None, grasp=None, mu0=None, sd_scale=1.0):
    rng = np.random.default_rng(seed)
    mu = (LO + HI) / 2
    mu[:6] = [110, -60, -60, 30, 20, 30]                         # start from a sensible overhand throw
    mu[6:13] = [0.0, 0.08, 0.16, 0.45, 0.35, 0.25, 0.35]
    if mu0 is not None:
        mu = np.clip(np.asarray(mu0, float), LO, HI)
    sd = (HI - LO) / 4 * sd_scale
    best = (-1e9, None)
    hist = []
    for it in range(iters):
        X = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), LO, HI)
        X[0] = mu
        jobs = [(x, bag, grasp) for x in X]
        S = np.array(pool.map(_score_job, jobs) if pool else [_score_job(j) for j in jobs])
        idx = np.argsort(-S)[:elite]
        if S[idx[0]] > best[0]:
            best = (float(S[idx[0]]), X[idx[0]].copy())
        mu = 0.7 * X[idx].mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * X[idx].std(0) + 0.3 * sd * 0.6, (HI - LO) * 0.01)
        hist.append(float(S[idx[0]]))
    return best, hist


def load_grasps():
    path = os.path.join(HERE, "assets", "grasps.npz")
    g = np.load(path)
    keys = ("qpos", "q_tgt", "pinch", "pinch_data", "phi_pinch")
    return [{k: g[k][i] for k in keys} for i in range(len(g["qpos"]))], g["rel"].reshape(len(g["qpos"]), -1, 3)


def _scatter_job(args):
    x, bag, grasp = args
    return rollout(x, bag, grasp=grasp)


def _timing_job(args):
    x, bag, grasp, dt = args
    y = x.copy()
    y[12] += dt
    return rollout(y, bag, grasp=grasp)["distance"]


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    t0 = time.time()
    grasps, rel = load_grasps()
    # a typical pickup: the one whose bag position in the gripper is closest to the median
    med = np.median(rel[:, 0], axis=0)
    typical = int(np.argmin(np.linalg.norm(rel[:, 0] - med, axis=1)))
    with mp.Pool(max(1, os.cpu_count())) as pool:
        plan = [("mini 2 in, 25 g", BAGS["mini 2 in, 25 g"], grasps[typical]),
                ("medium 3 in, 60 g", BAGS["medium 3 in, 60 g"], None)]
        for name, bag, grasp in plan:
            (s, x), hist = cem(bag, pool=pool, grasp=grasp)
            r = rollout(x, bag, grasp=grasp)
            results[name] = dict(bag=bag.__dict__, params=x.tolist(), start="table pickup",
                                 **r, history=hist)
            if not r["held"]:
                print(f"{name:26s} can't hold it through the wind-up", flush=True)
                continue
            print(f"{name:26s} {r['distance']:5.2f} m   release {r.get('release_speed', 0):4.2f} m/s at "
                  f"{r.get('release_angle', 0):5.1f} deg, h {r.get('release_height', 0):.2f} m   "
                  f"(from a table pickup, slipped {r['slipped']}, {time.time() - t0:.0f} s)", flush=True)
        # how much does the grip alone scatter the same throw?
        name, bag = "mini 2 in, 25 g", BAGS["mini 2 in, 25 g"]
        x = np.array(results[name]["params"])
        idx = np.random.default_rng(1).choice(len(grasps), 200, replace=False)
        rs = pool.map(_scatter_job, [(x, bag, grasps[i]) for i in idx])
        ok = [r for r in rs if r["held"] and r["released_at"] is not None]
        slips = sum(1 for r in ok if r["slipped"])
        xy = np.array([r["landed_xy"] for r in ok])
        drops = len(rs) - len(ok)
        sc = dict(n=len(rs), dropped=drops, slipped=slips, mean_xy=xy.mean(0).tolist(), std_xy_mm=(xy.std(0) * 1000).tolist(),
                  p90_radius_mm=float(np.percentile(np.linalg.norm(xy - xy.mean(0), axis=1), 90) * 1000))
        results["grasp_scatter"] = sc
        print(f"same throw from 200 different pickups: landing spread (1 sd) {sc['std_xy_mm'][0]:.0f} mm long x "
              f"{sc['std_xy_mm'][1]:.0f} mm across, 90 % within {sc['p90_radius_mm']:.0f} mm; "
              f"{slips} slipped out early, {drops} dropped", flush=True)
        # release timing
        base = results[name]["distance"]
        dts = (-0.010, -0.005, 0.005, 0.010)
        d = pool.map(_timing_job, [(x, bag, grasps[typical], dt) for dt in dts])
        results["release_timing"] = {f"{dt * 1000:+.0f} ms": float(v - base) for dt, v in zip(dts, d)}
        print("release timing:", {k: f"{v * 100:+.0f} cm" for k, v in results["release_timing"].items()}, flush=True)
    json.dump(results, open(os.path.join(OUT, "throw_study.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
