import numpy as np
from arm.arm import *
from arm.checks import interferences
B = build_static()
bad = 0
for q2 in np.arange(-10, 121, 10):
    q = dict(J1=0, J2=float(q2), J3=0, J4=0, J5=0, G=10)
    parts = [p for p in assemble(B, q) if p.body in ("turret", "upper")]
    dyn = {id(p) for p in parts if p.key.startswith(("cable_bal", "balancer_piston", "balancer_spring", "dowel_3x12", "dowel_3x8"))}
    def skip(a, b):
        return not (id(a) in dyn or id(b) in dyn or a.body != b.body)
    res = interferences(parts, min_vol=0.3, skip=skip)
    if res:
        bad += 1
        print(q2, [(r[0], r[1], r[2]) for r in res])
print("poses with clashes:", bad)
