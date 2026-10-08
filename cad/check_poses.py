# inter-body collision sweep over the range of motion (printed + actuator shells only)
import itertools, sys, time
import numpy as np
from arm.arm import *
from arm.params import JOINTS
B = build_static()
SHELL = lambda p: p.kind == "printed" or p.key.startswith("motor_bell") or p.key.startswith("cable")
poses = []
base = dict(J1=0.0, J2=45.0, J3=-60.0, J4=0.0, J5=0.0, G=20.0)
for jn in ("J2", "J3", "J4", "J5"):
    lo, hi = JOINTS[jn]["rom"]
    for a in (lo, hi):
        q = dict(base); q[jn] = float(a); poses.append(q)
for g in (0.0, 50.0):
    for r in (0.0, 90.0):
        q = dict(base); q["G"] = g; q["J5"] = r; poses.append(q)
for j4 in (-90.0, 70.0):
    for r in (-150.0, 150.0, 90.0):
        q = dict(base); q["J4"] = j4; q["J5"] = r; poses.append(q)
poses.append(dict(J1=150.0, J2=120.0, J3=-110.0, J4=70, J5=150, G=50))
poses.append(dict(J1=-150.0, J2=60.0, J3=110.0, J4=-90, J5=-150, G=0))
poses.append(dict(J1=0.0, J2=-10.0, J3=0.0, J4=0, J5=0, G=0))
order = ["base", "turret", "upper", "fore", "hand", "jawL", "jawR"]
for q in poses:
    parts = [p for p in assemble(B, q) if SHELL(p)]
    shapes = [(p, p.world_shape()) for p in parts]
    boxes = [s.BoundingBox() for _, s in shapes]
    hits = {}
    for i, j in itertools.combinations(range(len(parts)), 2):
        a, b = parts[i], parts[j]
        if a.body == b.body: continue
        if {a.key, b.key} == {"balancer_spool", "cable_bal_free"}:
            continue   # the cable is tied round the spool
        bi, bj = boxes[i], boxes[j]
        if bi.xmax < bj.xmin or bj.xmax < bi.xmin or bi.ymax < bj.ymin or bj.ymax < bi.ymin or bi.zmax < bj.zmin or bj.zmax < bi.zmin:
            continue
        v = shapes[i][1].intersect(shapes[j][1]).Volume()
        if v > 0.5:
            k = (a.body + ":" + a.key, b.body + ":" + b.key)
            hits[k] = max(hits.get(k, 0), round(v, 1))
    print({k: round(v) for k, v in q.items()}, "OK" if not hits else hits)
