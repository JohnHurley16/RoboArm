"""Checks capstan_ik against the CAD model and itself.

    python ik/test_ik.py          (from the repo root; needs cadquery for the CAD checks)
"""
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cad"))     # the CAD package (for the cross-checks)
import capstan_ik as K                                           # noqa: E402

rng = random.Random(1)
arm = K.Arm()
fails = 0


def check(name, cond, detail=""):
    global fails
    if not cond:
        fails += 1
        print(f"FAIL {name} {detail}")


def rand_q(margin=2.0):
    q = {k: rng.uniform(lo + margin, hi - margin) for k, (lo, hi) in K.LIMITS.items()}
    return q


# 1. constants and frames match the CAD ------------------------------------
try:
    from arm.arm import J2_Z, L1, L2, body_frames
    from arm.params import JOINTS, GRIP
    from arm import hand
    cad = True
except Exception as e:                                          # noqa: BLE001
    print("CAD not importable, skipping CAD checks:", e)
    cad = False

if cad:
    g = K.GEOM
    check("H_SHOULDER", g["H_SHOULDER"] == J2_Z, (g["H_SHOULDER"], J2_Z))
    check("L1/L2", (g["L1"], g["L2"]) == (L1, L2))
    check("XG", g["XG"] == hand.XG, (g["XG"], hand.XG))
    check("grip geom", (g["XP"], g["PZ"], g["LL"], g["FACE"]) == (hand.XP, hand.PZ, hand.LL, hand.FACE))
    for k in K.JOINTS:
        check(f"ratio {k}", abs(K.RATIO[k] - JOINTS[k]["ratio"]) < 1e-9)
        check(f"limits {k}", tuple(K.LIMITS[k]) == tuple(map(float, JOINTS[k]["rom"])))
    check("grip limits", tuple(K.LIMITS["G"]) == tuple(map(float, GRIP["phi"])))
    for phi in (0, 10, 25, 40, 50):
        check(f"opening {phi}", abs(K.phi_to_grip(phi) - hand.opening_mm(phi)) < 1e-9)
    worst_p = worst_a = 0.0
    for _ in range(500):
        q = rand_q(0)
        Fc = body_frames(q)
        Fk = arm.frames(q)
        for a_, b_ in (("upper", "shoulder"), ("fore", "elbow"), ("hand", "wrist"), ("grip", "grip")):
            worst_p = max(worst_p, np.abs(Fc[a_] - Fk[b_]).max())
        # finger frames: tool point is between the pads, jaw axis = grip z
        fu, fl = Fc["fu"][:3, 3], Fc["fl"][:3, 3]
        mid = (fu + fl) / 2 + Fc["grip"][:3, 0] * K.TCP_MODES["center"]
        worst_p = max(worst_p, np.abs(mid - Fk["tool"][:3, 3]).max())
        worst_a = max(worst_a, np.abs(Fc["grip"][:3, 2] - Fk["tool"][:3, 2]).max())
    check("frames vs CAD", worst_p < 1e-6 and worst_a < 1e-9, (worst_p, worst_a))
    print(f"CAD match: worst frame difference {worst_p:.2e} mm over 500 random poses")

# 2. FK -> IK -> FK round trips -------------------------------------------
n, n_ok, worst = 3000, 0, 0.0
for i in range(n):
    q = rand_q()
    if arm.check_keep_out(q):
        continue
    P = arm.fk(q)
    sols = arm.ik_all(P.x, P.y, P.z, P.pitch, P.roll, P.grip, seed=q)
    best = sols[0]
    # the original pose must be among the solutions (roll may differ by 180, same grip)
    match = None
    for s in sols:
        dj = max(abs(K._wrap(s.q[k] - q[k])) for k in ("J1", "J2", "J3", "J4"))
        d5 = min(abs(K._wrap(s.q["J5"] - q["J5"])), abs(K._wrap(s.q["J5"] - q["J5"] - 180)))
        if dj < 1e-6 and d5 < 1e-6:
            match = s
    check("original among solutions", match is not None, (q, [s.q for s in sols]))
    check("seeded best is the original", best is match, (q, best.q))
    Q = arm.fk(best.q)
    err = max(abs(Q.x - P.x), abs(Q.y - P.y), abs(Q.z - P.z))
    da = math.degrees(math.acos(min(1, max(-1, Q.T[:3, 0] @ P.T[:3, 0]))))
    dj = math.degrees(math.acos(min(1, abs(Q.T[:3, 2] @ P.T[:3, 2]))))     # jaws: either sign
    worst = max(worst, err)
    check("round trip", err < 1e-6 and da < 1e-4 and dj < 1e-4 and abs(Q.grip - P.grip) < 1e-9,
          (err, da, dj))
    n_ok += 1
print(f"round trips: {n_ok} poses, worst position error {worst:.2e} mm")

# 3. unreachable targets are reported --------------------------------------
try:
    arm.ik(900, 0, 100)
    check("unreachable raises", False)
except K.Unreachable as e:
    print("unreachable:", e)

# 4. motor mapping inverts ---------------------------------------------------
q = rand_q()
back = K.Arm.from_motors(K.Arm.to_motors(q))
check("motors invert", all(abs(back[k] - q[k]) < 1e-9 for k in (*K.JOINTS, "G")))

# 5. Jacobian vs finite position change -------------------------------------
q = {"J1": 20, "J2": 50, "J3": -70, "J4": -20, "J5": 10, "G": 20}
J = arm.jacobian(q)
dq = np.radians([0.01, -0.02, 0.015, 0.01, 0.02])
q2 = {k: q[k] + math.degrees(d) for k, d in zip(K.JOINTS, dq)}
q2["G"] = q["G"]
dp = arm.frames(q2)["tool"][:3, 3] - arm.frames(q)["tool"][:3, 3]
check("jacobian", np.abs(J[:3] @ dq - dp).max() < 1e-3, np.abs(J[:3] @ dq - dp).max())

# 6. straight line ---------------------------------------------------------
a = dict(x=300, y=-80, z=60, pitch=-60, roll=0, grip=30)
b = dict(x=300, y=80, z=60, pitch=-60, roll=90, grip=5)
path = arm.line(a, b, steps=40)
ends = arm.fk(path[-1].q)
check("line end", abs(ends.y - 80) < 1e-6 and abs(ends.grip - 5) < 1e-6)
print(f"line: {len(path)} steps, J1 {path[0].q['J1']:.1f} -> {path[-1].q['J1']:.1f} deg")

print("ALL PASS" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
