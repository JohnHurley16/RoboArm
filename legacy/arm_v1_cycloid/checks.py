"""Interference checking."""
import itertools
import cadquery as cq

# pairs that are supposed to overlap (threads, press fits modelled nominally)
EXPECTED = [
    ("m3_motor", "motor_base"), ("ins_", "shcs"), ("ins_", "bhcs"), ("ins_", "m3_"),
    ("grub", "act_output_carrier"), ("shcs_m2", "ins_m2"),
    ("ring_pin", "act_ring_housing"), ("ring_pin", "act_front_cap"),
    ("out_pin", "act_output_carrier"), ("motor_", "motor_"),
    ("act_eccentric_hub", "motor_shaft"), ("screw_m2", "joint_cap"), ("bhcs_m3x10", "jaw_left"),
    ("shcs_m3x16", "ins_m3"), ("bolt_m4", "ins_m4"), ("grub_m3x8", "shaft8"),
    ("screw_m2x20", "palm"), ("servo_spline", "servo_horn"), ("servo_horn", "link_drive_upper"),
    ("servo_spline", "link_drive_upper"), ("eclip", "shaft8"), ("nyloc", "shcs"),
    ("dowel_3x8", "upper_arm"), ("dowel_3x12", "balancer_piston"), ("nut_m3", "shcs_m3x20"),
    ("cable_bal_free", "dowel_3x8"), ("cable_bal_free", "balancer_spool"), ("balancer_spool", "cable_bal_free"), ("dowel_3x8", "balancer_spool"), ("cable_bal_free", "cable_bal_wrap"), ("cable_bal_wrap", "cable_bal_run"),
    ("brg_684", "shcs_m4"), ("shcs_m4", "nyloc_m4"), ("balancer_spacer", "shcs_m4"),
    ("cable_bal_inner", "balancer_piston_pulley"), ("cable_bal_inner", "cable_bal_inner"),
    ("cable_bal_run", "cable_bal_inner"), ("balancer_piston_pulley", "dowel_3x12"),
    ("screw_m2x20", "servo_body"), ("cable_bal_inner", "balancer_bracket"), ("cable_bal_run", "balancer_bracket"),
]
# heat-set inserts are melted into undersized holes: overlap with any printed part is fine
PRINTED_OK = ("ins_",)


def expected(a, b, pa=None, pb=None):
    if pa is not None:
        for pre in PRINTED_OK:
            if (a.startswith(pre) and pb.kind == "printed") or (b.startswith(pre) and pa.kind == "printed"):
                return True
    for x, y in EXPECTED:
        if (a.startswith(x) and b.startswith(y)) or (b.startswith(x) and a.startswith(y)):
            return True
    return False


def bbox_overlap(b1, b2, tol=0.0):
    return not (b1.xmax < b2.xmin - tol or b2.xmax < b1.xmin - tol or
                b1.ymax < b2.ymin - tol or b2.ymax < b1.ymin - tol or
                b1.zmax < b2.zmin - tol or b2.zmax < b1.zmin - tol)


def interferences(parts, min_vol=0.5, skip=None, groups=None):
    """Return list of (key_a, key_b, volume) for unexpected overlaps."""
    shapes = [(p.key, p.world_shape()) for p in parts]
    boxes = [s.BoundingBox() for _, s in shapes]
    out = []
    for i, j in itertools.combinations(range(len(shapes)), 2):
        ka, kb = shapes[i][0], shapes[j][0]
        if expected(ka, kb, parts[i], parts[j]) or (skip and skip(parts[i], parts[j])):
            continue
        if not bbox_overlap(boxes[i], boxes[j]):
            continue
        try:
            v = shapes[i][1].intersect(shapes[j][1]).Volume()
        except Exception as e:  # noqa
            v = -1
        if v > min_vol or v < 0:
            out.append((ka, kb, round(v, 2), i, j))
    return out
