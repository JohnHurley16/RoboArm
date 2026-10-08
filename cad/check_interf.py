import sys, time
from arm.arm import *
from arm.checks import interferences
B = build_static()
qname = sys.argv[1] if len(sys.argv) > 1 else "home"
q = {"home": HOME, "pose": POSE}.get(qname)
if q is None:
    q = eval(qname)
parts = assemble(B, q)
# skip pairs inside the same actuator module (validated separately) -> mark actuator parts
act_keys = {"act_hub","act_rotor_adapter","act_spider_tpu","act_bearing_spacer","act_magnet_plug","act_rear_plate","act_rear_lid","drv_pcb","drv_parts","drv_chip","magnet_drv","motor_screw","bhcs_m3x5","shcs_m3x10","act_motor_can","act_ring_housing","act_front_cap","act_eccentric_hub","act_cycloid_disc","act_output_carrier",
            "act_retainer_ring","motor_base","motor_bell","motor_stator","motor_shaft","motor_wires",
            "m3_motor","brg_6800","brg_6801","brg_6705","ring_pin","out_pin","grub_m3x6","bhcs_m3x6","enc_board","shcs_m2x5","ins_m2","act_end_cap","shcs_m3x10","ins_m3"}
def _ak(k):
    return k in act_keys or k.startswith(("act_", "motor_", "drv_"))
def skip(a, b):
    return _ak(a.key) and _ak(b.key) and a.body == b.body
t = time.time()
res = interferences(parts, min_vol=0.5, skip=skip)
print("time", round(time.time()-t,1), "n", len(res))
for ka, kb, v, i, j in res:
    print(f"{ka:28s} [{parts[i].body:6s}] x {kb:28s} [{parts[j].body:6s}] {v}")
