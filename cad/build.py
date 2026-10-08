"""Regenerate everything:  python build.py  [--no-step] [--no-viewer]

Outputs (in ./out):
  stl/            one STL per printed part, already in print orientation
  step/           full assembly STEP (all printed + purchased parts, coloured),
                  actuator module STEP, and every printed part as its own STEP
  BOM.md / BOM.csv
  report.md       ratios, torque estimates, cable lengths, print checks
"""
import argparse
import base64
import collections
import json
import math
import os
import sys

import numpy as np
import cadquery as cq

from arm.params import MOTORS, DRIVER, CAP, JOINTS, STACK, GRIP
from arm.core import mat_loc
from arm.arm import build_static, assemble, POSE, HOME, body_frames, cable_length, J2_Z
from arm.actuator import actuator_parts
from arm.render import mesh_of

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rot_to_down(v):
    """Rotation matrix taking unit vector v to (0,0,-1)."""
    a = np.asarray(v, float)
    a /= np.linalg.norm(a)
    b = np.array([0, 0, -1.0])
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    d = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if d > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - d) / s ** 2)


def print_oriented(part):
    M = np.eye(4)
    M[:3, :3] = rot_to_down(part.print_down)
    s = part.local_shape().moved(mat_loc(M))
    bb = s.BoundingBox()
    M2 = np.eye(4)
    M2[:3, 3] = (-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin)
    return s.moved(mat_loc(M2))


def overhang_report(shape):
    """Area of down-facing surfaces steeper than 45 deg that are not on the bed."""
    V, F = mesh_of(shape, 0.1, 0.2)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    zc = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
    steep = (nz < -math.cos(math.radians(45))) & (zc > 0.3)
    bed = (nz < -0.99) & (zc <= 0.3)
    return float(area[steep].sum()), float(area[bed].sum()), float(area.sum())


def balancer_cable_length():
    from arm import balancer as BL
    f = BL.fitted()
    longest_path = max(BL.path_length(q) for q in range(-10, 121, 5))
    run = BL.idler_xz()[0] - BL.BULK_X[1]
    inner = 2 * (f["spring"]["free"] - f["defl_min"] + BL.BULK_X[1] - BL.BULK_X[0] - BL.PP_X) + math.pi * BL.R_PP
    return longest_path + run + inner + 2 * 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "stl"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "step", "printed_parts"), exist_ok=True)

    print("building model ...")
    bodies = build_static()
    parts = assemble(bodies, POSE)

    # ----------------------------------------------------------- printed ----
    printed = collections.OrderedDict()
    qty = collections.Counter()
    for p in parts:
        if p.kind == "printed":
            qty[p.key] += 1
            printed.setdefault(p.key, p)
    checks = []
    for key, p in printed.items():
        s = print_oriented(p)
        cq.exporters.export(cq.Workplane().add(s), os.path.join(OUT, "stl", f"{key}.stl"),
                            tolerance=0.02, angularTolerance=0.1)
        if not args.no_step:
            cq.exporters.export(cq.Workplane().add(p.local_shape()),
                                os.path.join(OUT, "step", "printed_parts", f"{key}.step"))
        bb = s.BoundingBox()
        oh, bed, tot = overhang_report(s)
        vol = p.local_shape().Volume() / 1000.0
        checks.append(dict(key=key, qty=qty[key], size=(bb.xlen, bb.ylen, bb.zlen), vol=vol,
                           overhang=oh, bed=bed, valid=p.local_shape().isValid()))
        print(f"  {key:24s} x{qty[key]:<3d} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f}  "
              f"overhang {oh:7.0f} mm2  bed {bed:6.0f} mm2")

    # -------------------------------------------------------------- BOM ------
    hwq = collections.Counter(p.bom for p in parts if p.bom)
    with open(os.path.join(OUT, "BOM.csv"), "w") as f:
        f.write("type,item,qty\n")
        for key, n in qty.items():
            f.write(f"printed,{key},{n}\n")
        for item, n in sorted(hwq.items()):
            f.write(f"purchased,\"{item}\",{n}\n")
        for jn, j in JOINTS.items():
            f.write(f"purchased,\"Dyneema 1 mm cable for {jn} ({cable_length(j):.0f} mm incl. tails)\",1\n")
        f.write(f"purchased,\"Dyneema 1 mm cable for shoulder balancer ({balancer_cable_length():.0f} mm incl. tails)\",1\n")

    total_g = sum(c["vol"] * c["qty"] for c in checks) * 1.27 * 0.45  # PETG, ~45% effective fill
    lines = ["# Bill of materials", "", "## Printed parts", "",
             "| Part | Qty | Bounding box (print orientation, mm) | Overhang >45° (mm²) |",
             "|---|---:|---|---:|"]
    for c in checks:
        sx, sy, sz = c["size"]
        lines.append(f"| `{c['key']}.stl` | {c['qty']} | {sx:.0f} × {sy:.0f} × {sz:.0f} | {c['overhang']:.0f} |")
    lines += ["", f"Estimated filament: ~{total_g:.0f} g PETG (4 walls, 25-35 % gyroid).", "",
              "## Purchased parts", "", "| Item | Qty |", "|---|---:|"]
    for item, n in sorted(hwq.items()):
        lines.append(f"| {item} | {n} |")
    for jn, j in JOINTS.items():
        lines.append(f"| Dyneema / UHMWPE braid Ø{CAP['cable_d']:g} mm, {jn}: {cable_length(j):.0f} mm (incl. 2×40 mm tails) | 1 |")
    lines.append(f"| Dyneema / UHMWPE braid Ø{CAP['cable_d']:g} mm, shoulder balancer: {balancer_cable_length():.0f} mm (incl. tails) | 1 |")
    open(os.path.join(OUT, "BOM.md"), "w").write("\n".join(lines) + "\n")

    # ----------------------------------------------------------- report -----
    try:
        tq = json.load(open(os.path.join(OUT, "torque.json")))
    except Exception:
        tq = None
    rep = ["# Design report", "",
           "Drive: gimbal motor → zero-backlash TPU jaw coupling → capstan (no gearbox). "
           f"Each joint has a {DRIVER['name']} driver on the back of its motor, reading the motor's own encoder "
           "and the joint encoder on its aux port.", "",
           "| Joint | Motor | Capstan ratio | Rated / peak at the joint (η≈0.92) | Gravity, worst pose (0 / 200 / 500 g payload) | Range of motion | Cable |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for jn, j in JOINTS.items():
        m = MOTORS[j["motor"]]
        g = ""
        if tq and tq["joints"][jn].get("gravity"):
            gv = tq["joints"][jn]["gravity"]
            g = f"{gv['0']:.2f} / {gv['200']:.2f} / {gv['500']:.2f} Nm"
        rep.append(f"| {jn} {j['name']} | {m['name']} | {j['ratio']:.1f}:1 | "
                   f"{m['rated_nm'] * j['ratio'] * 0.92:.2f} / {m['peak_nm'] * j['ratio'] * 0.92:.2f} Nm | {g or '–'} | "
                   f"{j['rom'][0]}° … {j['rom'][1]}° | {cable_length(j):.0f} mm |")
    if tq:
        rep += ["", f"Moving mass (everything past the shoulder): {tq['moving_mass_g']:.0f} g. "
                "The shoulder gravity figures are without the balancer."]
    from arm import hand as H
    from arm.arm import L1, L2
    sv = GRIP["servo"]
    lo, hi = GRIP["phi"]
    grip_n = sv["stall_nm"] / (H.LL / 1000.0)
    rep += ["", f"Shoulder height: {J2_Z:.0f} mm. Upper arm {L1:.0f} mm, forearm {L2:.0f} mm, "
            f"wrist pitch axis to fingertip ≈ {H.XG + H.XP + H.LL + H.FIN_TIP:.0f} mm.", "",
            f"Gripper: {sv['name']} driving a synced parallelogram. Pad gap {H.opening_mm(lo):.1f} mm "
            f"(closed) to {H.opening_mm(hi):.1f} mm (open), before Fin Ray deflection. "
            f"Squeeze at stall ≈ {grip_n:.1f} N per finger (servo stall torque / link length).", ""]
    rep += report_balancer() + report_accuracy()
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(rep) + "\n")

    # ------------------------------------------------------------ STEP ------
    if not args.no_step:
        print("writing STEP assemblies ...")
        for name, q in (("arm_assembly", POSE), ("arm_assembly_home", HOME)):
            asm = cq.Assembly(name=name)
            for i, p in enumerate(assemble(bodies, q)):
                asm.add(p.local_shape(), name=f"{p.body}__{p.key}__{i}", loc=mat_loc(p.M),
                        color=cq.Color(*p.color))
            asm.save(os.path.join(OUT, "step", f"{name}.step"))
        for mk in sorted({j["motor"] for j in JOINTS.values()}):
            asm = cq.Assembly(name=f"actuator_{mk}")
            for i, p in enumerate(actuator_parts(mk)):
                asm.add(p.local_shape(), name=f"{p.key}__{i}", loc=mat_loc(p.M), color=cq.Color(*p.color))
            asm.save(os.path.join(OUT, "step", f"actuator_{mk}.step"))

    # ----------------------------------------------------------- viewer -----
    if not args.no_viewer:
        print("exporting viewer meshes ...")
        export_viewer(bodies)
    print("done")


def report_balancer():
    from arm import balancer as BL
    f = BL.fitted()
    sp = f["spring"]
    return ["## Shoulder balancer", "",
            "Zero-length cable spring: ear on the upper arm → idler on the turret → 2:1 spring cartridge behind the turret.", "",
            f"| | Without balancer | With balancer |", "|---|---:|---:|",
            f"| Peak shoulder torque (any pose) | {f['max_before']:.2f} Nm | {f['max_after']:.2f} Nm |",
            f"| RMS shoulder torque (over all poses) | {f['rms_before']:.2f} Nm | {f['rms_after']:.2f} Nm |",
            f"| Motor heating (∝ torque², RMS) | 1× | {(f['rms_after'] / f['rms_before']) ** 2:.2f}× |", "",
            f"Cable tension {f['T_min']:.0f}–{f['T_max']:.0f} N over the shoulder's range (1 mm Dyneema ≈ 1 kN break).", "",
            f"Spring: music wire {sp['wire']:g} mm, OD {sp['od']:g} mm, free length ≈ {math.ceil(sp['free'])} mm, "
            f"rate ≈ {sp['rate']:.1f} N/mm, {sp['total']:.1f} total coils, ends closed & ground. "
            f"Working stress {sp['tau']:.0f} MPa (limit {sp['allow']:.0f}). "
            f"Deflection {f['defl_min']:.1f}–{f['defl_max']:.1f} mm, solid length {sp['solid']:.1f} mm.", "",
            f"The ear has three pin holes ({' / '.join(f'{b:g}' for b in BL.B_HOLES)} mm). Middle is nominal; "
            "move the pin out if the arm sags, in if it floats up. Each hole step is about ±6 % balance.", ""]


def report_accuracy():
    p = os.path.join(OUT, "accuracy.json")
    if not os.path.exists(p):
        return []
    a = json.load(open(p))
    rows = ["## Position accuracy", "",
            "There is no gearbox, so there is no gear backlash. The error budget is coupling slack (zero when new), "
            "cable stretch under load, and the encoders.", "",
            "| Joint | Cable stiffness | Stretch under worst gravity load | Coupling slack if the spider takes a 0.05 mm set | Motor-encoder step at the joint |",
            "|---|---:|---:|---:|---:|"]
    for j, r in a["joints"].items():
        rows.append(f"| {j} | {r['K']:.0f} Nm/rad | {r['defl']:.3f}° | {r['slack_worn']:.3f}° | {r['motor_enc']:.4f}° |")
    rows += ["", "| Pose | Reach | Backlash (new) | Slack if worn (worst / RSS) | Cable stretch | Motor-encoder step |",
             "|---|---:|---:|---:|---:|---:|"]
    for pn, pv in a["poses"].items():
        rows.append(f"| {pn} | {pv['reach']:.0f} mm | 0 | ±{pv['worn_worst']:.2f} / ±{pv['worn_rss']:.2f} mm | "
                    f"{pv['defl_worst']:.2f} mm | {pv['res_worst']:.3f} mm |")
    e = a["encoder_limited_reach"]
    k = sorted(e, key=float)
    rows += ["", "Controlling J1–J4 on their joint encoders removes coupling slack and cable stretch from the static "
             "position. The error is then set by the joint encoders (arm straight out, worst / RSS):", "",
             f"- 14-bit resolution: ±{e[k[0]]['worst']:.1f} / ±{e[k[0]]['rss']:.1f} mm (repeatability)",
             f"- 0.1° after calibration: ±{e[k[1]]['worst']:.1f} / ±{e[k[1]]['rss']:.1f} mm",
             f"- 0.3° uncalibrated: ±{e[k[2]]['worst']:.1f} / ±{e[k[2]]['rss']:.1f} mm", "",
             f"Cable stiffness assumes an effective EA of {a['assumptions']['EA_N'] / 1000:.0f} kN for 1 mm Dyneema braid "
             "(an estimate; real braid varies and creeps).", ""]
    return rows


def export_viewer(bodies):
    """Shared mesh library (quantised) + per-body instances, for the web viewer."""
    lib, lib_index = [], {}
    out = {"lib": lib, "bodies": {}}
    for bname, plist in bodies.items():
        inst = []
        for p in plist:
            if p.kind == "hardware" and p.key.startswith(("ins_", "shim", "washer", "grub", "screw_m2", "shcs_m2")):
                continue  # hidden / tiny: skip in the viewer only
            key = (id(p.shape), tuple(round(c, 3) for c in p.color))
            if key not in lib_index:
                tol = 0.2 if p.kind == "printed" else 0.4
                V, F = mesh_of(p.local_shape(), tol, 0.5 if p.kind == "printed" else 0.8)
                lo, hi = V.min(0), V.max(0)
                scale = np.maximum(hi - lo, 1e-6) / 65535.0
                Q = np.round((V - lo) / scale).astype(np.uint16)
                idx = F.astype(np.uint16 if len(V) < 65536 else np.uint32)
                lib_index[key] = len(lib)
                lib.append(dict(c=list(key[1]), lo=lo.round(4).tolist(), s=scale.tolist(),
                                v=base64.b64encode(Q.tobytes()).decode(),
                                f=base64.b64encode(idx.tobytes()).decode(), i32=bool(len(V) >= 65536),
                                name=p.key, kind=p.kind))
            inst.append([lib_index[key]] + [round(float(x), 4) for x in p.M.T.ravel()])
        out["bodies"][bname] = inst
    # ---- balancer (pose-dependent) meshes
    from arm import balancer as BL
    from arm.core import Part as _P

    def add_lib(shape, color, kind, name):
        V, F = mesh_of(shape, 0.25, 0.6)
        lo, hi = V.min(0), V.max(0)
        scale = np.maximum(hi - lo, 1e-6) / 65535.0
        Q = np.round((V - lo) / scale).astype(np.uint16)
        lib.append(dict(c=[round(c, 3) for c in color], lo=lo.round(4).tolist(), s=scale.tolist(),
                        v=base64.b64encode(Q.tobytes()).decode(),
                        f=base64.b64encode(F.astype(np.uint16 if len(V) < 65536 else np.uint32).tobytes()).decode(),
                        i32=bool(len(V) >= 65536), name=name, kind=kind))
        return len(lib) - 1

    fit = BL.fitted()
    sp = fit["spring"]
    s_ref = sp["free"]
    dyn = {}
    pis, pul = BL.piston(), BL.piston_pulley()
    from arm.core import frame as _frame, COLORS as _C
    from arm import hardware as _hw
    dyn["piston"] = [
        [add_lib(pis.val(), BL.C_A, "printed", "balancer_piston")] + np.eye(4).T.ravel().round(4).tolist(),
        [add_lib(pul.val(), BL.C_G, "printed", "balancer_piston_pulley")] +
        _frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (BL.PP_X, -1.5, 0)).T.ravel().round(4).tolist(),
    ]
    dyn["spring"] = [add_lib(BL.spring_solid(s_ref, sp["od"], sp["wire"], sp["total"] - 0.5).val(),
                             (0.62, 0.64, 0.68), "hardware", "balancer_spring")] + \
        BL._ax_frame((BL.BULK_X[0], BL.PLANE_Y, BL.pod_axis_z()), (-1, 0, 0)).T.ravel().round(4).tolist()
    ax, az = BL.idler_xz()
    out["bal"] = dict(A=[ax, az], r=BL.R_IDL, beta=BL.BETA, b=BL.B_R, y=BL.PLANE_Y, bulk=list(BL.BULK_X),
                      pod_z=BL.pod_axis_z(), strand_z=BL.strand_z(), anchor_z=BL.anchor_z(), pp_x=BL.PP_X,
                      free=sp["free"], s_ref=s_ref, L0=fit["L0"], k=fit["k_eff"], dyn=dyn,
                      cable_c=list(_C["cable"]))
    from arm.arm import L1, L2
    from arm import hand as H
    out["kin"] = dict(J2_Z=J2_Z, L1=L1, L2=L2, XG=H.XG, XP=H.XP, PZ=H.PZ, LH=H.LH, LL=H.LL,
                      phi=list(GRIP["phi"]), open_max=round(H.opening_mm(GRIP["phi"][1]), 1),
                      face=GRIP["face"])
    out["rom"] = {k: list(v["rom"]) for k, v in JOINTS.items()}
    out["ratio"] = {k: round(v["ratio"], 1) for k, v in JOINTS.items()}
    out["names"] = {k: v["name"] for k, v in JOINTS.items()}
    path = os.path.join(OUT, "viewer_data.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    viewer = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer", "viewer_data.json")
    if os.path.isdir(os.path.dirname(viewer)):
        import shutil
        shutil.copy(path, viewer)                       # the 3D viewer loads it from beside index.html
    print("  viewer data", round(os.path.getsize(path) / 1e6, 2), "MB,", len(lib), "meshes")


if __name__ == "__main__":
    main()
