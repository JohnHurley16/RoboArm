"""Lay every printed part out onto Bambu Lab P1S plates (256 x 256 mm),
in assembly order, so you can build one sub-assembly while the next prints.

Parts are split into STAGES (actuators first, then base/turret, upper arm,
forearm, wrist and gripper).  Each stage is packed one filament per plate;
spare room on a plate is topped up with small parts from the next stage, so
nothing ever arrives later than it is needed.  Writes
out/plates/plate_NN_s<stage>_<colour>.3mf (standard 3MF; Bambu Studio keeps
the positions), a preview of every plate and out/plates/PLATES.md.

    python make_plates.py          (run python build.py first)
"""
import collections
import math
import os
import zipfile

import numpy as np
import trimesh

from arm.arm import build_static, assemble, POSE
from arm.core import COLORS

BED = 256.0
MARGIN = 8.0          # keep clear of the plate edge
GAP = 6.0             # between parts
PLATE_BUDGET_G = 160.0   # ≈ 11 h per plate, so one failed print never costs a whole day
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "plates")
STL = os.path.join(HERE, "out", "stl")

GROUPS = [   # name, filament, colour test
    ("white", "PETG, white", lambda c: abs(c[0] - COLORS["printed"][0]) < 0.02 and c[2] > 0.8),
    ("graphite", "PETG, graphite / dark grey", lambda c: abs(c[0] - COLORS["printed2"][0]) < 0.02),
    ("orange", "PETG, orange", lambda c: abs(c[0] - COLORS["accent"][0]) < 0.02),
    ("tpu", "TPU 95A, black (external spool, not the AMS)", lambda c: c[0] < 0.15),
]
SMALL_G = 6.0            # parts this light may ride along with one oversize part (base, turret)
SUPPORTS = {"act_hub"}            # parts that need tree supports
SOLO_SUPPORT_PLATE = False        # True = supported parts on a plate of their own


# ------------------------------------------------------------- packing -----
class MaxRects:
    """MaxRects bin packing, best-short-side-fit, with 90 deg rotation."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.free = [(0.0, 0.0, w, h)]
        self.used = []

    def insert(self, w, h):
        best = None
        for fx, fy, fw, fh in self.free:
            for rw, rh, rot in ((w, h, False), (h, w, True)):
                if rw <= fw + 1e-6 and rh <= fh + 1e-6:
                    ss = min(fw - rw, fh - rh)
                    ls = max(fw - rw, fh - rh)
                    key = (ss, ls)
                    if best is None or key < best[0]:
                        best = (key, fx, fy, rw, rh, rot)
        if best is None:
            return None
        _, x, y, rw, rh, rot = best
        self._split((x, y, rw, rh))
        self.used.append((x, y, rw, rh))
        return x, y, rot

    def _split(self, r):
        x, y, w, h = r
        new = []
        for f in self.free:
            fx, fy, fw, fh = f
            if x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy:
                new.append(f)
                continue
            if x > fx:
                new.append((fx, fy, x - fx, fh))
            if x + w < fx + fw:
                new.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy:
                new.append((fx, fy, fw, y - fy))
            if y + h < fy + fh:
                new.append((fx, y + h, fw, fy + fh - y - h))
        pruned = []
        for i, a in enumerate(new):
            contained = False
            for j, b in enumerate(new):
                if i != j and a[0] >= b[0] - 1e-9 and a[1] >= b[1] - 1e-9 and \
                        a[0] + a[2] <= b[0] + b[2] + 1e-9 and a[1] + a[3] <= b[1] + b[3] + 1e-9:
                    if a != b or i > j:
                        contained = True
                        break
            if not contained and a[2] > 1 and a[3] > 1:
                pruned.append(a)
        self.free = pruned


def est_grams(mesh, group):
    if group == "tpu":
        return mesh.volume / 1000.0 * 1.21 * 0.9
    return mesh.volume / 1000.0 * 1.27 * 0.6


def pack(items, area, group):
    """items: list of (name, mesh).  Returns list of plates, each a list of
    (name, mesh, x, y, rotated).  Respects the bed size and a mass budget."""
    order = sorted(items, key=lambda it: -(it[1].extents[0] + GAP) * (it[1].extents[1] + GAP))
    plates = []
    for name, mesh in order:
        w, h = mesh.extents[0] + GAP, mesh.extents[1] + GAP
        g = est_grams(mesh, group)
        placed = False
        for pl in sorted(plates, key=lambda p: -p["g"]):
            solo_big = len(pl["items"]) == 1 and pl["g"] > PLATE_BUDGET_G
            if pl["g"] + g > PLATE_BUDGET_G and pl["items"] and not (solo_big and g <= SMALL_G):
                continue
            r = pl["bin"].insert(w, h)
            if r:
                pl["items"].append((name, mesh) + r)
                pl["g"] += g
                placed = True
                break
        if not placed:
            b = MaxRects(area, area)
            r = b.insert(w, h)
            if r is None:
                raise SystemExit(f"{name} does not fit on a {BED:.0f} mm plate")
            plates.append(dict(bin=b, items=[(name, mesh) + r], g=g, early=set()))
    return plates


def top_up(plates, items, group):
    """Fit extra (name, mesh) items into existing plates within the budget.
    Returns the items that did not fit."""
    left = []
    for name, mesh in sorted(items, key=lambda it: -(it[1].extents[0] + GAP) * (it[1].extents[1] + GAP)):
        w, h = mesh.extents[0] + GAP, mesh.extents[1] + GAP
        g = est_grams(mesh, group)
        for pl in sorted(plates, key=lambda p: p["g"]):
            if pl["g"] + g > PLATE_BUDGET_G:
                continue
            r = pl["bin"].insert(w, h)
            if r:
                pl["items"].append((name, mesh) + r)
                pl["early"].add(len(pl["items"]) - 1)
                pl["g"] += g
                break
        else:
            left.append((name, mesh))
    return left


# --------------------------------------------------------------- 3MF -------
def write_3mf(path, placed, title):
    objs, items = [], []
    ids = {}
    for name, mesh, M in placed:
        if name not in ids:
            oid = len(ids) + 1
            ids[name] = oid
            v = mesh.vertices
            f = mesh.faces
            vs = "".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>' for a, b, c in v)
            ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
            objs.append(f'<object id="{oid}" name="{name}" type="model"><mesh><vertices>{vs}</vertices>'
                        f'<triangles>{ts}</triangles></mesh></object>')
        t = " ".join(f"{M[r, c]:.6f}" for c in range(4) for r in range(3))
        items.append(f'<item objectid="{ids[name]}" transform="{t}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<metadata name="Title">{title}</metadata><resources>' + "".join(objs) + '</resources><build>'
             + "".join(items) + '</build></model>')
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)


# ------------------------------------------------------------ preview ------
def preview(plates_out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    n = len(plates_out)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.5))
    axs = np.atleast_1d(axs).ravel()
    fills = {"white": "#e9e9e5", "graphite": "#3a3d44", "orange": "#ef6a1c", "tpu": "#1f2024"}
    for ax in axs:
        ax.axis("off")
    for ax, pl in zip(axs, plates_out):
        ax.add_patch(plt.Rectangle((0, 0), BED, BED, fill=True, fc="#d4d7dc", ec="#888", lw=1))
        polys = []
        for name, mesh, M in pl["placed"]:
            mm = mesh.copy()
            mm.apply_transform(M)
            tri = mm.vertices[mm.faces][:, :, :2]
            polys.append(tri)
        allp = np.concatenate(polys)
        ax.add_collection(PolyCollection(allp, facecolors=fills[pl["group"]], edgecolors="none"))
        ax.set_xlim(-4, BED + 4)
        ax.set_ylim(-4, BED + 4)
        ax.set_aspect("equal")
        ax.set_title(f"Plate {pl['n']} · stage {pl['stage']}: {pl['group']}" + (" (supports)" if pl["supports"] else ""),
                     fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ------------------------------------------------------------- stages ------
# Each stage: what to print, then what to build while the next stage prints.
# parts: {group: {key: qty}}.  top_up: fill spare room with next-stage parts.
G, O, W, TPU = "graphite", "orange", "white", "tpu"
STAGES = [
    dict(title="Pathfinder actuator: J2 shoulder (GL60)", top_up=False,
         parts={O: {"act_hub": 1, "act_rotor_adapter_GL60": 1, "act_retainer_ring": 1,
                    "act_bearing_spacer": 1, "act_magnet_plug_GL60": 1},
                G: {"act_front_cap": 1, "act_motor_can_GL60": 1, "act_rear_plate_GL60": 1, "act_rear_lid_GL60": 1},
                TPU: {"act_spider_tpu": 5, "finray_tpu": 2}},
         gate="This is the one time to let the printer sit idle for an hour. Before you start plate {next}, "
              "press both 6705 bearings into the front cap and slide the hub through them. They should go in "
              "with firm thumb pressure, and the hub should spin with no rock. The spider should need a light "
              "push to seat on the dogs. If anything is loose or won't go in, adjust `FIT` in `arm/params.py`, "
              "run `python build.py` and `python make_plates.py`, and reprint plates 1–2 before going on. "
              "Every other actuator part uses the same fits.",
         build=[
             "Finish the actuator (README, *Actuator*, steps 1–6).",
             "Wire its moteus-c1 to the 24 V supply and the CAN-FD adapter. Set its CAN ID to 2 "
             "(`conf set id.id 2`, then `conf write`).",
             "Calibrate it with the motor free to spin (`moteus_tool -t 2 --calibrate`), before any cable is wound.",
             "Spin it in position mode, then turn the hub by hand with the power off. It should be smooth, with "
             "no click at reversal. Check that the motor-encoder and aux-encoder readings agree.",
         ],
         note="All the TPU is on this stage, so the external spool only has to be loaded once."),
    dict(title="The other four actuators: J3 (GL60), J1, J4, J5 (GL40)", top_up=True,
         parts={O: {"act_hub": 4, "act_rotor_adapter_GL60": 1, "act_rotor_adapter_GL40": 3,
                    "act_retainer_ring": 4, "act_bearing_spacer": 4, "act_magnet_plug_GL60": 1,
                    "act_magnet_plug_GL40": 3},
                G: {"act_front_cap": 4, "act_motor_can_GL60": 1, "act_rear_plate_GL60": 1, "act_rear_lid_GL60": 1,
                    "act_motor_can_GL40": 3, "act_rear_plate_GL40": 3, "act_rear_lid_GL40": 3}},
         build=[
             "Build the remaining four actuators exactly like the first. Label them: the GL60 is J3; the GL40s are "
             "J1, J4 and J5.",
             "Set CAN IDs 1, 3, 4 and 5, then calibrate each one free-spinning.",
             "Daisy-chain all five drivers on one CAN bus and confirm every ID answers.",
         ]),
    dict(title="Base and turret (J1 yaw)", top_up=True,
         parts={G: {"base": 1, "j1_spacer": 1, "joint_cap": 1},
                W: {"turret_body": 1, "joint_cap": 1},
                O: {"turret_drum": 1}},
         build=[
             "Fit the heat-set inserts in the base and turret.",
             "Press the 608s into the base boss, mount the J1 actuator, and fit the J1 encoder and its graphite "
             "joint cap under the base.",
             "Fit the 8 mm shaft, `j1_spacer`, `turret_drum` and `turret_body`. Wind and tension the J1 cable.",
             "Mount the J2 actuator in the turret cheek and fit the J2 encoder with its white joint cap.",
             "Test J1: sweep the turret through its range under power and check the cable doesn't ride up the drum.",
         ]),
    dict(title="Upper arm and shoulder balancer (J2)", top_up=True,
         parts={W: {"upper_arm": 1, "joint_cap": 1, "balancer_tube": 1},
                G: {"upper_arm_cheek_a": 1, "upper_arm_cheek_b": 1, "balancer_bracket": 1,
                    "balancer_spacer": 1, "balancer_piston_pulley": 1},
                O: {"sector_j2": 1, "balancer_idler": 1, "balancer_spool": 1, "balancer_piston": 1}},
         build=[
             "Bolt `sector_j2` to the upper arm and fit the J2 shaft into the turret.",
             "Wind and tension the J2 cable. Test the shoulder through its range with a low current limit.",
             "Bolt on both elbow cheeks with the J3 actuator, and fit the J3 encoder with its joint cap.",
             "Build and fit the balancer (README, *Shoulder balancer*), then do its motors-off drift test.",
         ]),
    dict(title="Forearm (J3 elbow)", top_up=True,
         parts={W: {"forearm": 1, "joint_cap": 1},
                G: {"forearm_cheek_a": 1, "forearm_cheek_b": 1},
                O: {"sector_j3": 1}},
         build=[
             "Bolt `sector_j3` to the forearm and fit it into the elbow.",
             "Wind and tension the J3 cable and test the elbow.",
             "Bolt on the wrist cheeks with the J4 actuator, and fit the J4 encoder with its joint cap.",
         ]),
    dict(title="Wrist and gripper (J4 pitch, J5 roll)", top_up=False,
         parts={W: {"hand": 1, "palm": 1, "servo_cover": 1, "finger": 2},
                G: {"wrist_yoke": 1, "roll_pulley_guard": 1, "roll_spacer": 1, "link_drive_upper": 1,
                    "link_drive_lower": 1, "link_follow": 2},
                O: {"sector_j4": 1, "roll_flange": 1}},
         build=[
             "Bolt `sector_j4` to the hand, fit it into the wrist and wind the J4 cable.",
             "Build the wrist roll and gripper (README, *Wrist roll* and *Gripper*), using the Fin Ray tips "
             "from stage 1.",
             "Full-arm test: home every joint, then run the range-of-motion sweep slowly with low current limits.",
         ]),
]
PLATE_ORDER = [O, G, W, TPU]   # within a stage: short orange plates first


# --------------------------------------------------------------- main ------
def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    parts = assemble(build_static(), POSE)
    qty = collections.Counter()
    for p in parts:
        if p.kind == "printed":
            gname = next(g[0] for g in GROUPS if g[2](p.color))
            qty[(p.key, gname)] += 1
    # every printed part must be in exactly one stage
    staged = collections.Counter()
    for st in STAGES:
        for g, d in st["parts"].items():
            for k, n in d.items():
                staged[(k, g)] += n
    if staged != qty:
        miss = qty - staged
        extra = staged - qty
        raise SystemExit(f"stage lists out of date.\n  missing: {dict(miss)}\n  extra: {dict(extra)}")
    meshes = {k: trimesh.load(os.path.join(STL, f"{k}.stl")) for k, _ in qty}
    fil = {g[0]: g[1] for g in GROUPS}
    area = BED - 2 * MARGIN

    # items still to place, per stage and filament
    todo = [{g: [(k, meshes[k]) for k, n in d.items() for _ in range(n)] for g, d in st["parts"].items()}
            for st in STAGES]
    plates_out = []
    for si, st in enumerate(STAGES):
        for g in PLATE_ORDER:
            items = todo[si].get(g, [])
            if not items:
                continue
            pls = pack(items, area, g)
            if st["top_up"] and si + 1 < len(STAGES) and todo[si + 1].get(g):
                todo[si + 1][g] = top_up(pls, todo[si + 1][g], g)
            for pl in pls:
                placed = []
                for idx, (name, mesh, x, y, rot) in enumerate(pl["items"]):
                    M = np.eye(4)
                    if rot:
                        M[:3, :3] = trimesh.transformations.rotation_matrix(math.pi / 2, (0, 0, 1))[:3, :3]
                    b = mesh.bounds.copy()
                    corners = np.array([[b[0][0], b[0][1], 0, 1], [b[1][0], b[1][1], 0, 1],
                                        [b[0][0], b[1][1], 0, 1], [b[1][0], b[0][1], 0, 1]]) @ M.T
                    mn = corners[:, :2].min(0)
                    M[0, 3] = MARGIN + x + GAP / 2 - mn[0]
                    M[1, 3] = MARGIN + y + GAP / 2 - mn[1]
                    M[2, 3] = -b[0][2]
                    placed.append((name, mesh, M, idx in pl["early"]))
                # centre the layout on the bed (better heat and first layer for small plates)
                lo, hi = np.full(2, 1e9), np.full(2, -1e9)
                for name, mesh, M, _ in placed:
                    v = trimesh.transform_points(mesh.bounds[[0, 1]] * [[1, 1, 0]], M)
                    c = trimesh.transform_points(np.array([[a, b, 0] for a in mesh.bounds[:, 0]
                                                          for b in mesh.bounds[:, 1]]), M)[:, :2]
                    lo, hi = np.minimum(lo, c.min(0)), np.maximum(hi, c.max(0))
                shift = BED / 2 - (lo + hi) / 2
                for _, _, M, _ in placed:
                    M[:2, 3] += shift
                plates_out.append(dict(stage=si + 1, group=g, filament=fil[g], supports=any(
                    n in SUPPORTS for n, *_ in placed), placed=placed))

    by_stage = collections.defaultdict(list)
    t = 0.0
    for i, pl in enumerate(plates_out, 1):
        pl["n"] = i
        fn = f"plate_{i:02d}_s{pl['stage']}_{pl['group']}{'_supports' if pl['supports'] else ''}.3mf"
        write_3mf(os.path.join(OUT, fn), [(n, m, M) for n, m, M, _ in pl["placed"]], f"Capstan Arm 5 plate {i}")
        grams = sum(est_grams(m, pl["group"]) for _, m, _, _ in pl["placed"])
        hours = grams / 14.0 + 0.3          # + heat-up / levelling
        t += hours
        now = collections.Counter(n for n, _, _, e in pl["placed"] if not e)
        early = collections.Counter(n for n, _, _, e in pl["placed"] if e)
        fmt = lambda c: ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in sorted(c.items()))
        names = fmt(now) + (f"; *early, for stage {pl['stage'] + 1}:* {fmt(early)}" if early else "")
        pl.update(fn=fn, grams=grams, hours=hours, done_at=t, names=names)
        by_stage[pl["stage"]].append(pl)
    preview([dict(pl, placed=[(n, m, M) for n, m, M, _ in pl["placed"]]) for pl in plates_out],
            os.path.join(OUT, "plates_overview.png"))

    tot_g = sum(pl["grams"] for pl in plates_out)
    md = ["# Print plates in assembly order (Bambu Lab P1S, 256 × 256 mm)", "",
          "The plates are numbered in the order to print them. They're grouped into **stages**: when a stage's "
          "last plate comes off the printer, start the next plate and build that stage while it prints. "
          "Each stage's build takes a few hours, and the next stage's plates usually take longer than that, so "
          "the printer is rarely waiting on you.", "",
          "The most technical work comes first: the actuators. Stage 1 is one complete actuator, so you can "
          "check bearing fits, the coupling and the driver setup on a single unit before printing the other four.", "",
          "Each plate is one filament. Some plates carry a few small parts from the next stage in their spare "
          "room (marked *early*), so nothing arrives late and no near-empty plates are needed.", "",
          "Open each `plate_NN_*.3mf` in Bambu Studio. The parts keep their positions and orientations. Pick the "
          "filament named in the table, slice, print.", "",
          "## Timeline", "",
          "| Stage | Plates | Printer time | Build while the next stage prints |", "|---:|---|---:|---|"]
    for si, st in enumerate(STAGES, 1):
        pls = by_stage[si]
        h = sum(p["hours"] for p in pls)
        md.append(f"| {si} | {pls[0]['n']}–{pls[-1]['n']} | ≈ {h:.0f} h (done at ≈ {pls[-1]['done_at']:.0f} h) "
                  f"| {st['title']} |" if len(pls) > 1 else
                  f"| {si} | {pls[0]['n']} | ≈ {h:.0f} h (done at ≈ {pls[-1]['done_at']:.0f} h) | {st['title']} |")
    md += ["", f"Total ≈ {tot_g:.0f} g of filament and ≈ {t:.0f} h of printing. The gram and hour figures are "
           "rough (about 60 % effective fill, about 14 g/h, plus heat-up); Bambu Studio's slice preview gives the "
           "real numbers.", ""]
    for si, st in enumerate(STAGES, 1):
        md += [f"## Stage {si}: {st['title']}", "",
               "| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |", "|---:|---|---|---|---:|---:|---|"]
        for pl in by_stage[si]:
            md.append(f"| {pl['n']} | `{pl['fn']}` | {pl['filament']} | {'tree (auto)' if pl['supports'] else 'none'} "
                      f"| {pl['grams']:.0f} | {pl['hours']:.1f} | {pl['names']} |")
        if st.get("note"):
            md += ["", st["note"]]
        nxt = f"While stage {si + 1} prints" if si < len(STAGES) else "Once the last plate is off"
        if st.get("gate"):
            md += ["", "**Fit check, before starting plate " f"{by_stage[si + 1][0]['n']}:** "
                   + st["gate"].format(next=by_stage[si + 1][0]["n"])]
        md += ["", f"**{nxt}:**", ""] + [f"{j}. {b}" for j, b in enumerate(st["build"], 1)] + [""]
    md += ["## Settings", "",
           "- PETG: textured PEI plate, 0.2 mm layers, 4 walls, 30 % gyroid. Open the lid or remove the top glass.",
           "- 100 % infill for `act_hub` and `act_rotor_adapter_*`: select them and set infill per object.",
           "- Plates marked 'tree (auto)': enable tree supports (auto). Only the `act_hub` flanges need them.",
           "- `base` is the longest print, about a day at these settings. 3 walls and 15 % gyroid are plenty "
           "for it and save several hours. Start it in the evening.",
           "- TPU 95A: feed from the external spool holder (not the AMS), about 3–4 mm³/s. Fin Ray tips: 2 walls, "
           "0 % infill. Spiders: 100 % infill.",
           "- Install heat-set inserts after printing, with the parts cool.",
           "- To change the order, edit `STAGES` in `make_plates.py` and rerun it. It refuses to run if a "
           "printed part is missing from the stages or listed twice.", ""]
    open(os.path.join(OUT, "PLATES.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
