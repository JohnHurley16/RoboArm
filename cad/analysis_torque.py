"""Joint torque budget: gravity load (worst pose, with and without a payload)
against each motor's rated / peak torque through its capstan ratio."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # paths below are relative to cad/
_os.makedirs("out", exist_ok=True)
import json, math
import numpy as np
from arm.arm import build_static, body_frames
from arm.params import JOINTS, MOTORS
from arm.massprops import body_props, gravity_torque
from analysis_accuracy import tcp

ETA = 0.92          # capstan + bearing efficiency (estimate)
B = build_static()
P = body_props(B)
moving = ["upper", "fore", "hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"]
out = dict(moving_mass_g=sum(P[b][0] for b in moving), joints={})
try:
    bal = json.load(open("out/balancer.json"))
except Exception:
    bal = None
for j in JOINTS:
    m = MOTORS[JOINTS[j]["motor"]]
    r = JOINTS[j]["ratio"]
    row = dict(motor=m["name"], ratio=r, rated=m["rated_nm"] * r * ETA, peak=m["peak_nm"] * r * ETA)
    if j in ("J2", "J3", "J4"):
        worst = {}
        for pay in (0, 200, 500):
            mx = 0.0
            for q2 in range(-10, 121, 10):
                for q3 in range(-110, 111, 20):
                    for q4 in (-90, -45, 0, 45, 70):
                        q = dict(J1=0, J2=q2, J3=q3, J4=q4, J5=0, G=20)
                        t = gravity_torque(P, body_frames(q), j, pay, tcp(q))
                        mx = max(mx, abs(t))
            worst[pay] = mx
        row["gravity"] = worst
    out["joints"][j] = row
json.dump(out, open("out/torque.json", "w"), indent=1)
print("moving mass", round(out["moving_mass_g"]), "g")
for j, r in out["joints"].items():
    g = r.get("gravity")
    print(j, r["motor"], f"{r['ratio']:.1f}:1 rated {r['rated']:.2f} Nm peak {r['peak']:.2f} Nm",
          "" if not g else "gravity worst " + ", ".join(f"{k} g payload: {v:.2f}" for k, v in g.items()))
