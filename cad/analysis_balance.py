"""Fit the balancer (cable rate + preload) to the arm's gravity torque."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # paths below are relative to cad/
_os.makedirs("out", exist_ok=True)
import json, math
import numpy as np
from arm.arm import build_static, body_frames
from arm import balancer as BL
from arm.massprops import body_props, gravity_torque_j2
from arm.params import JOINTS

B = build_static()
P = body_props(B)
q2s = np.arange(-10, 121, 5)
q3s = np.arange(-110, 111, 10)
q4s = (-90, -45, 0, 45, 70)
G = np.array([[[gravity_torque_j2(P, body_frames(dict(J1=0, J2=a, J3=b, J4=c, J5=0, G=20)))
                for c in q4s] for b in q3s] for a in q2s])
L = np.array([BL.path_length(a) for a in q2s])
geo = [BL.tangent(a) for a in q2s]
arm = []
for (Pp, Tp, n, _) in geo:
    u = (Tp - Pp) / n
    r = (Pp - np.array([0, BL.J2Z()])) / 1000.0
    arm.append(r[0] * u[1] - r[1] * u[0])
arm = np.array(arm)


def residual(k, L0):
    Tn = np.maximum(0, k * (L - L0))
    tau = arm * Tn
    return G + tau[:, None, None], Tn


best = None
for k in np.linspace(0.3, 1.6, 131):
    for L0 in np.linspace(-20, 25, 91):
        r, Tn = residual(k, L0)
        if Tn.min() < 4.0:          # keep the cable taut everywhere
            continue
        rms = math.sqrt((r ** 2).mean())
        if best is None or rms < best[0]:
            best = (rms, k, L0, abs(r).max(), Tn.min(), Tn.max())
rms, k, L0, mx, tmin, tmax = best
k_s = 4 * k                                  # 2:1 tackle
d_spring = (L - L0) / 2                       # spring deflection from free
sd = BL.spring_design(k_s, d_spring.max(), 2 * tmax)
straight = G[list(q2s).index(0), list(q3s).index(0), list(q4s).index(0)]
out = dict(k_eff=k, L0=L0, rms_after=rms, max_after=mx, rms_before=float(np.sqrt((G ** 2).mean())),
           max_before=float(abs(G).max()), straight=float(straight), T_min=tmin, T_max=tmax,
           spring=sd, defl_min=float(d_spring.min()), defl_max=float(d_spring.max()),
           curve=dict(q2=q2s.tolist(), tension=(k * (L - L0)).round(1).tolist(),
                      g_straight=G[:, list(q3s).index(0), list(q4s).index(0)].round(3).tolist(),
                      spring=(arm * k * (L - L0)).round(3).tolist()))
json.dump(out, open("out/balancer.json", "w"), indent=1)
json.dump(out, open("arm/balancer_fit.json", "w"), indent=1)
print(json.dumps({k_: v for k_, v in out.items() if k_ != "curve"}, indent=1))
