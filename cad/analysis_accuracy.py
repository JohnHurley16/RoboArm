"""End-effector position error for the direct capstan design.

There is no gearbox, so the error budget is:
  1. coupling slack: the TPU jaw coupling is preloaded (zero backlash when
     new); 'worn' case assumes the spider takes a 0.05 mm set per side.
     Slack at the motor is divided by the capstan ratio at the joint.
  2. cable stretch under the joint's gravity load (Dyneema, both strands
     preloaded).  Effective EA of 1 mm braid is an estimate (see EA_N).
  3. encoder limits (resolution, and accuracy before / after calibration).
Joint errors are mapped to the tool point with the arm's Jacobian.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # paths below are relative to cad/
_os.makedirs("out", exist_ok=True)
import json
import math
import numpy as np

from arm.params import JOINTS, CAP
from arm.arm import body_frames
from arm import hand as H

EA_N = 15000.0          # effective axial stiffness of 1 mm Dyneema braid, N (estimate)
WRAP_EFF = 10.0         # extra effective cable length from the wrap zones, mm (estimate)
DOG_R = 10.5            # mean radius of the coupling dogs, mm
SET_MM = 0.05           # spider set per side in the 'worn' case, mm
ENC_RES_DEG = 360 / 2 ** 14


def tcp(q):
    F = body_frames(q)
    x = H.XP + H.LL * math.cos(math.radians(q["G"])) + (H.FIN_BASE[0] + H.FIN_TIP) / 2
    return (F["grip"] @ np.array([x, 0, 0, 1.0]))[:3]


def jacobian(q):
    J = {}
    for j in ("J1", "J2", "J3", "J4", "J5"):
        d = 1e-3
        qp, qm = dict(q), dict(q)
        qp[j] += d
        qm[j] -= d
        J[j] = (tcp(qp) - tcp(qm)) / (2 * math.radians(d))      # mm per rad
    return J


def joint_stiffness(j):
    """N m / rad at the joint from cable stretch."""
    jj = JOINTS[j]
    Rc, rp, d = jj["Rc"], CAP["pulley_r"], jj["d"]
    span = math.sqrt(max(d ** 2 - (Rc - rp) ** 2, 1.0))
    L = (span + WRAP_EFF) / 1000.0
    k_strand = EA_N / L
    return 2 * k_strand * (Rc / 1000.0) ** 2, span


POSES = {
    "Reach (arm straight out)": dict(J1=0, J2=0, J3=0, J4=0, J5=0, G=20),
    "Table pick (elbow down)": dict(J1=0, J2=35, J3=-95, J4=-30, J5=0, G=20),
    "Close in (folded)": dict(J1=0, J2=80, J3=-110, J4=-40, J5=0, G=20),
}


def main():
    try:
        tq = json.load(open("out/torque.json"))["joints"]
    except Exception:
        tq = {}
    try:
        bal = json.load(open("out/balancer.json"))
    except Exception:
        bal = None
    joints = {}
    for j in JOINTS:
        K, span = joint_stiffness(j)
        slack = math.degrees(2 * SET_MM / DOG_R) / JOINTS[j]["ratio"]
        load = 0.0
        if j in ("J3", "J4") and j in tq:
            load = tq[j]["gravity"]["0"]
        if j == "J2" and bal:
            load = bal["max_after"]
        joints[j] = dict(K=K, span=span, slack_worn=slack, load=load,
                         defl=math.degrees(load / K) if K else 0.0,
                         motor_enc=ENC_RES_DEG / JOINTS[j]["ratio"])
    poses = {}
    for pn, q in POSES.items():
        Jm = jacobian(q)
        lever = {j: float(np.linalg.norm(Jm[j])) for j in Jm}
        rad = math.radians
        worn = {j: lever[j] * rad(joints[j]["slack_worn"] / 2) for j in Jm}
        defl = {j: lever[j] * rad(joints[j]["defl"]) for j in Jm}
        res = {j: lever[j] * rad(joints[j]["motor_enc"]) for j in Jm}
        poses[pn] = dict(reach=float(np.linalg.norm(tcp(q)[:2])), lever=lever,
                         new_backlash=0.0,
                         worn_worst=sum(worn.values()), worn_rss=math.sqrt(sum(v * v for v in worn.values())),
                         defl_worst=sum(defl.values()), res_worst=sum(res.values()))
    enc = {}
    Jm = jacobian(POSES["Reach (arm straight out)"])
    for acc in (ENC_RES_DEG, 0.1, 0.3):
        v = [np.linalg.norm(Jm[j]) * math.radians(acc) for j in ("J1", "J2", "J3", "J4")]
        enc[round(acc, 3)] = dict(worst=float(sum(v)), rss=float(math.sqrt(sum(x * x for x in v))))
    out = dict(joints=joints, poses=poses, encoder_limited_reach=enc,
               assumptions=dict(EA_N=EA_N, WRAP_EFF=WRAP_EFF, SET_MM=SET_MM, DOG_R=DOG_R))
    json.dump(out, open("out/accuracy.json", "w"), indent=1, default=float)
    for j, r in joints.items():
        print(f"{j}: stiffness {r['K']:.0f} Nm/rad, worn slack {r['slack_worn']:.3f} deg, "
              f"load {r['load']:.2f} Nm -> {r['defl']:.3f} deg, motor-encoder step {r['motor_enc']:.4f} deg")
    for pn, p in poses.items():
        print(pn, f"reach {p['reach']:.0f}: backlash new 0, worn ±{p['worn_worst']:.2f} (RSS {p['worn_rss']:.2f}) mm; "
                  f"cable stretch {p['defl_worst']:.2f} mm; motor-encoder step {p['res_worst']:.3f} mm")
    print("joint-encoder limited:", {k: (round(v['worst'], 2), round(v['rss'], 2)) for k, v in enc.items()})


if __name__ == "__main__":
    main()
