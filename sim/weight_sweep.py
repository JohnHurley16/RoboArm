"""Max throw vs bag weight: floppy 3 in bag picked off the table, best open-loop
throw found by the optimizer at each weight, with the motor and grip limits in.

    python sim/weight_sweep.py   -> sim/out/weight_sweep.json
"""
import json, math, multiprocessing as mp, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import throw_study as T
from skills import Sim, pick
from scene import Bag
import pinch as P, control as C

READY = np.radians([0.0, 95.0, -75.0, -40.0, 0.0])


def grasp_for(bag):
    """One real pickup + carry to the ready pose; returns a grasp state (or None)."""
    s = Sim(bag, bag_pose=(0.22, 0.03, 15))
    ok, _ = pick(s)
    if not ok:
        return None, "couldn't pick it up"
    s.move_joints(READY, 1.1)
    s.settle(0.3)
    if s.pinched < 0:
        return None, "slipped out while carrying" if s.slipped else "dropped while carrying"
    return dict(qpos=s.d.qpos.copy(), q_tgt=s.q_tgt.copy(), pinch=np.array([s.pinched]),
                pinch_data=s.m.eq_data[s.pinch_eq[s.pinched]].copy(), phi_pinch=np.array([s.phi_pinch])), "ok"


def main():
    masses = [0.025, 0.060, 0.100, 0.150, 0.250, 0.350, 0.450]
    out = []
    t0 = time.time()
    hold = P.hold_force(np, 0.0, C.SERVO["stall_nm"])
    warm = json.load(open(os.path.join(T.OUT, "throw_study.json")))["mini 2 in, 25 g"]["params"]
    with mp.Pool(os.cpu_count()) as pool:
        for mass in masses:
            bag = Bag(size=0.075, thick=0.020, mass=mass)
            g, why = grasp_for(bag)
            row = dict(mass_g=mass * 1000, hold_g=hold / 9.81 / mass)
            if g is None:
                row.update(distance=None, note=why)
            else:
                # two searches: warm-started from the best throw so far, and a fresh one; keep the better
                best = None
                for mu0, sc, seed in ((warm, 0.5, 1), (None, 1.0, 2)):
                    (score, x), _ = T.cem(bag, iters=30, pop=48, pool=pool, grasp=g, mu0=mu0, sd_scale=sc, seed=seed)
                    if best is None or score > best[0]:
                        best = (score, x)
                x = best[1]
                warm = x
                r = T.rollout(x, bag, grasp=g)
                row.update(distance=r["distance"] if r["released_at"] is not None else None,
                           release_speed=r.get("release_speed"), release_angle=r.get("release_angle"),
                           slipped=bool(r["slipped"]), note="slipped out mid-swing" if r["slipped"] else "released by the gripper")
            out.append(row)
            d = row["distance"]
            print(f"{mass * 1000:5.0f} g: " + (f"{d:4.2f} m, release {row['release_speed']:.2f} m/s, {row['note']}" if d else row["note"])
                  + f"   (pinch holds {row['hold_g']:.0f} g of acceleration; {time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "weight_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
