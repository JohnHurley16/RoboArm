"""Desk-sized, multi-colour display model of the whole arm (default 3 in tall, about 1:4.2).

The outside matches the real arm in the display pose; the inside is solid.
Every purchased part is either kept as a colour detail (screw heads, shaft
ends, washers) or melted into the solid interior.  The model is cut into four
pieces that print without fighting each other and glue together on 1.75 mm
filament dowels:

    A  base                         C  elbow  (upper-arm end + J3 + forearm start)
    B  turret + shoulder + balancer D  wrist and gripper

Writes out/mini/capstan_arm_mini.3mf (one Bambu Studio project, four
multi-colour objects, filament per part already assigned: 1 white, 2 graphite,
3 orange, 4 black), per-colour STLs as a fallback, renders and MINI.md.

    python make_mini.py [scale]        (default: 3 in tall)
"""
import math
import os
import sys
import uuid
import zipfile

import numpy as np
import trimesh
import manifold3d as mf

from arm.arm import build_static, assemble, body_frames, POSE
from arm.core import COLORS
from arm.render import mesh_of, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "mini")

TARGET_HEIGHT = 76.2     # mm: 3 in tall as it stands
FULL_HEIGHT = 317.37     # mm: full-size arm in the display pose
SCALE = TARGET_HEIGHT / FULL_HEIGHT
DOWEL_D = 1.75          # filament used as glue dowels
DOWEL_HOLE = 2.0        # hole for it (mini mm)
DOWEL_DEPTH = 4.0       # each side
BED = 256.0

SLOTS = ["white", "graphite", "orange", "black"]          # AMS slot / filament 1..4
SLOT_RGB = {"white": (0.92, 0.92, 0.90), "graphite": (0.19, 0.20, 0.23),
            "orange": (0.96, 0.42, 0.10), "black": (0.07, 0.07, 0.08)}
SLOT_HEX = {"white": "#EBEBE6", "graphite": "#30333A", "orange": "#F56B1A", "black": "#121214"}
PRIORITY = ["black", "orange", "white"]                     # graphite = everything else

# pieces: which bodies go where; links that are cut, at x (mm, full size,
# measured from the proximal joint along the link) between the sector and the
# next fork's cheeks.
PIECES = [
    ("A_base", "Base", {"base"}),
    ("B_shoulder", "Turret, shoulder and balancer", {"turret"}),
    ("C_elbow", "Elbow", set()),
    ("D_wrist", "Wrist and gripper", {"hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"}),
]
SPLITS = {"upper": ("upper_arm", 49.0, "B_shoulder", "C_elbow"),
          "fore": ("forearm", 36.0, "C_elbow", "D_wrist")}
DROP = {"shaft8_j1"}
TO_BASE = {"j1_spacer"}       # rides on the base, so the turret prints standing on its drum
SLIVER = 0.02                 # mm^3: boolean crumbs smaller than this are dropped

# per-object slicer settings written into the Bambu project (keys from Bambu Studio's PrintConfig)
_SUP = {"enable_support": "1", "support_type": "tree(auto)", "support_filament": "2",
        "support_interface_filament": "2"}
OBJECT_SETTINGS = {
    "B_shoulder": _SUP,
    "C_elbow": dict(_SUP, brim_type="outer_only", brim_width="4"),
    "D_wrist": dict(_SUP, brim_type="outer_only", brim_width="5"),
}          # yaw shaft: replaced by a dowel so the turret glues onto the base


def slot_of(p):
    c = np.asarray(p.color)
    if p.kind == "printed":
        return min(("white", "printed"), ("graphite", "printed2"), ("orange", "accent"),
                   ("black", None), key=lambda s: np.linalg.norm(c - (np.asarray(COLORS[s[1]]) if s[1]
                                                                    else np.array([0.13, 0.14, 0.16]))))[0]
    if p.kind == "hardware":
        if np.allclose(c, COLORS["screw"], atol=0.02):
            return "black"            # screw heads, nuts
        if np.allclose(c, COLORS["steel"], atol=0.03) or np.allclose(c, COLORS["magnet"], atol=0.03) \
                or c.mean() > 0.85:
            return "white"            # shaft ends, washers, shims, servo horn
    return "graphite"                 # hidden: motors, drivers, bearings, inserts, spring


# ------------------------------------------------------------ geometry ------
def to_manifold(V, F):
    m = mf.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                tri_verts=np.ascontiguousarray(F, np.uint32))
    M = mf.Manifold(m)
    if M.status() != mf.Error.NoError:
        m.merge()
        M = mf.Manifold(m)
    if M.status() != mf.Error.NoError:
        raise RuntimeError(M.status())
    return M


def to_trimesh(M):
    m = M.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def union(lst):
    lst = [m for m in lst if not m.is_empty()]
    if not lst:
        return mf.Manifold()
    return mf.Manifold.batch_boolean(lst, mf.OpType.Add)


def fill_cavities(M):
    """Close internal voids: every inside-out shell of the mesh is a cavity.
    Works on the index buffer directly (no sub-mesh copies) to keep memory low."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    m = M.to_mesh()
    V = np.asarray(m.vert_properties, np.float64)[:, :3]
    F = np.asarray(m.tri_verts, np.int64)
    nf, nv = len(F), len(V)
    # faces sharing a vertex are linked (cavity shells never share vertices with the outside)
    rows = np.repeat(np.arange(nf), 3)
    A = sp.csr_matrix((np.ones(nf * 3, np.int8), (rows, F.ravel())), shape=(nf, nv))
    nc, lab = connected_components(sp.bmat([[None, A], [A.T, None]]).tocsr(), directed=False)
    flab = lab[:nf]
    vol6 = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]]))
    shell_vol = np.bincount(flab, weights=vol6) / 6.0
    cav = []
    for c in np.nonzero(shell_vol < -1e-6)[0]:
        f = F[flab == c][:, ::-1]
        used, inv = np.unique(f, return_inverse=True)
        try:
            cav.append(to_manifold(V[used], inv.reshape(-1, 3)))
        except RuntimeError:
            pass
    del m, V, F, A
    return union([M] + cav) if cav else M


def cylinder(p0, p1, d, seg=32):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    c = mf.Manifold.cylinder(L, d / 2, d / 2, seg)
    z = (p1 - p0) / L
    x = np.cross(z, [0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    A = np.column_stack([x, y, z, p0])
    return c.transform(A.tolist())


def polylabel_points(section_polys, r_min):
    """One dowel point per cross-section island thick enough to take it."""
    from shapely.ops import polylabel
    pts = []
    for poly in section_polys:
        p = polylabel(poly, tolerance=0.02)
        if poly.exterior.distance(p) >= r_min and all(i.distance(p) >= r_min for i in poly.interiors):
            pts.append((p.x, p.y))
    return pts


# --------------------------------------------------------------- build ------
def build(scale):
    print("assembling ...")
    bodies = build_static()
    F = body_frames(POSE)
    parts = assemble(bodies, POSE)
    S = np.diag([scale, scale, scale, 1.0])
    Fs = {k: S @ v @ np.diag([1 / scale] * 3 + [1]) for k, v in F.items()}   # frames in mini space
    cache = {}
    groups = {p[0]: {s: [] for s in SLOTS} for p in PIECES}
    cut_tool = {}
    print("meshing", len(parts), "parts ...")
    for p in parts:
        if p.kind == "cable" or p.key in DROP:
            continue
        k = (p.key, id(p.shape))
        if k not in cache:
            V, Fc = mesh_of(p.local_shape(), tol=0.05, ang=0.2)
            cache[k] = (V, Fc)
        V, Fc = cache[k]
        Vw = (V @ p.M[:3, :3].T + p.M[:3, 3]) * scale
        if np.linalg.det(p.M[:3, :3]) < 0:
            Fc = Fc[:, ::-1]
        M = to_manifold(Vw, Fc)
        slot = slot_of(p)
        if p.body in SPLITS:
            link, xc, prox, dist = SPLITS[p.body]
            Fb = Fs[p.body]
            n = Fb[:3, 0]
            off = float(n @ Fb[:3, 3] + xc * scale)
            cut_tool[p.body] = (n, off)
            if p.key == link:
                hi = M.trim_by_plane(n.tolist(), off)
                lo = M.trim_by_plane((-n).tolist(), -off)
                groups[dist][slot].append(hi)
                groups[prox][slot].append(lo)
            else:
                cx = float(n @ (Vw.mean(0)))
                groups[dist if cx > off else prox][slot].append(M)
        elif p.key in TO_BASE:
            groups["A_base"][slot].append(M)
        else:
            piece = next(pc[0] for pc in PIECES if p.body in pc[2])
            groups[piece][slot].append(M)
    # fill the hollow underside of the base (the real one is a shell, printed upside down)
    from arm.arm import BASE_R, Hb, TP
    groups["A_base"]["graphite"].append(mf.Manifold.cylinder((Hb - TP - 1.5) * scale, (BASE_R - 4.2) * scale,
                                                             (BASE_R - 4.2) * scale, 128))
    return groups, cut_tool, Fs


def finish(groups, cut_tool, Fs, scale):
    """Union, fill, dowel holes, colour partition.  Returns {piece: {slot: Manifold}}."""
    res = {}
    for name, title, _ in PIECES:
        U = {s: union(groups[name][s]) for s in SLOTS}
        groups[name] = None
        solid = fill_cavities(union(list(U.values())))
        # keep the main body, drop specks that float free
        comps = sorted(solid.decompose(), key=lambda m: -m.volume())
        keep = [c for c in comps if c.volume() > 0.5]
        dropped = sum(c.volume() for c in comps if c.volume() <= 0.5)
        solid = union(keep)
        res[name] = dict(solid=solid, U=U, dropped=dropped, n_comp=len(keep),
                         comp_vols=[round(c.volume(), 1) for c in keep])
    # dowel holes: J1 axis between A and B, and at each link cut
    holes = {p[0]: [] for p in PIECES}
    j1_top = None
    a_top = res["A_base"]["solid"].bounding_box()[5]
    b_bot = res["B_shoulder"]["solid"].bounding_box()[2]
    for pc, z0, z1 in (("A_base", a_top - DOWEL_DEPTH * 2, a_top + 1), ("B_shoulder", b_bot - 1, b_bot + DOWEL_DEPTH * 2)):
        holes[pc].append(cylinder((0, 0, z0), (0, 0, z1), DOWEL_HOLE))
    dowel_log = [("A–B", "J1 axis", 1, DOWEL_DEPTH * 2.5)]
    for body, (n, off) in cut_tool.items():
        _, _, prox, dist = SPLITS[body]
        Fb = Fs[body]
        # cross-section of the distal piece on the cut plane
        tm = to_trimesh(res[dist]["solid"])
        o = n * off
        sec = tm.section(plane_origin=o, plane_normal=n)
        if sec is None:
            sec = tm.section(plane_origin=o + n * 0.02, plane_normal=n)
        p2, T2 = sec.to_2D()
        pts = polylabel_points(p2.polygons_full, DOWEL_HOLE / 2 + 0.45)
        for x, y in pts:
            w = (T2 @ [x, y, 0, 1])[:3]
            holes[dist].append(cylinder(w - n * 1, w + n * DOWEL_DEPTH, DOWEL_HOLE))
            holes[prox].append(cylinder(w + n * 1, w - n * DOWEL_DEPTH, DOWEL_HOLE))
        dowel_log.append((f"{prox[0]}–{dist[0]}", body, len(pts), DOWEL_DEPTH * 2 - 0.5))
    out = {}
    for name, title, _ in PIECES:
        r = res[name]
        solid = r["solid"] - union(holes[name]) if holes[name] else r["solid"]
        U = r["U"]
        taken = mf.Manifold()
        regions = {}
        for s in PRIORITY:
            reg = (solid ^ U[s]) - taken
            regions[s] = reg
            taken = taken + U[s]
        regions["graphite"] = solid - taken
        for k, reg in regions.items():
            regions[k] = union([c for c in reg.decompose() if c.volume() > SLIVER])
        out[name] = dict(regions=regions, solid=solid, dropped=r["dropped"], n_comp=r["n_comp"])
    return out, dowel_log


# ---------------------------------------------------------- orientation -----
def print_orientation(name, solid, cut_tool):
    """Rotation (3x3) that puts the piece's best face on the bed."""
    if name in ("A_base", "B_shoulder"):
        return np.eye(3)
    body = {"C_elbow": "upper", "D_wrist": "fore"}[name]
    n, off = cut_tool[body]
    # cut face (normal pointing back toward the parent) goes down: -n -> -z, i.e. n -> +z
    z = n / np.linalg.norm(n)
    x = np.cross([0, 1, 0], z) if abs(z[1]) < 0.9 else np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def overhang_area(tm, Rm):
    nz = tm.face_normals @ Rm[2]
    zc = (tm.triangles_center @ Rm[2])
    low = zc > zc.min() + 0.3
    return float(tm.area_faces[(nz < -0.707) & low].sum())


def best_spin(tm, Rm):
    """Spin about z so the footprint is smallest (packs better)."""
    best = None
    for a in range(0, 180, 5):
        Rz = trimesh.transformations.rotation_matrix(math.radians(a), (0, 0, 1))[:3, :3]
        v = tm.vertices @ (Rz @ Rm).T
        e = v.max(0) - v.min(0)
        if best is None or e[0] * e[1] < best[0]:
            best = (e[0] * e[1], Rz @ Rm)
    return best[1]


# ---------------------------------------------------------------- 3MF -------
def mesh_xml(tm):
    vs = "".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>' for a, b, c in tm.vertices)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tm.faces)
    return f"<mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh>"


def write_bambu_3mf(path, objects, title):
    """objects: list of (name, [(part_name, slot, trimesh)], 4x4 placement).
    Bambu Studio project layout: parts in 3D/Objects/object_N.model, assembled
    by components in 3D/3dmodel.model, filament per part in
    Metadata/model_settings.config."""
    ns = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
          'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
          'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"')
    U = lambda: str(uuid.uuid4())
    files, rels, main_objs, build_items, cfg_objs, cfg_inst, asm = {}, [], [], [], [], [], []
    next_id = 1
    for oi, (oname, parts, P) in enumerate(objects, 1):
        fpath = f"/3D/Objects/object_{oi}.model"
        sub, comps, cfg_parts = [], [], []
        for pname, slot, tm in parts:
            pid = next_id
            next_id += 1
            sub.append(f'<object id="{pid}" p:UUID="{U()}" type="model">{mesh_xml(tm)}</object>')
            comps.append(f'<component p:path="{fpath}" objectid="{pid}" p:UUID="{U()}" '
                         f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
            cfg_parts.append(f'    <part id="{pid}" subtype="normal_part">\n'
                             f'      <metadata key="name" value="{pname}"/>\n'
                             f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                             f'      <metadata key="extruder" value="{SLOTS.index(slot) + 1}"/>\n'
                             f'    </part>')
        files[fpath[1:]] = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {ns}>'
                            f'<metadata name="BambuStudio:3mfVersion">1</metadata><resources>'
                            + "".join(sub) + "</resources><build/></model>")
        rels.append(f'<Relationship Target="{fpath}" Id="rel-{oi}" '
                    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>')
        oid = next_id
        next_id += 1
        main_objs.append(f'<object id="{oid}" p:UUID="{U()}" type="model"><components>{"".join(comps)}'
                         f'</components></object>')
        t = " ".join(f"{P[r, c]:.6f}" for c in range(4) for r in range(3))
        build_items.append(f'<item objectid="{oid}" p:UUID="{U()}" transform="{t}" printable="1"/>')
        extra = "".join(f'    <metadata key="{k}" value="{v}"/>\n' for k, v in OBJECT_SETTINGS.get(oname, {}).items())
        cfg_objs.append(f'  <object id="{oid}">\n    <metadata key="name" value="{oname}"/>\n'
                        f'    <metadata key="extruder" value="2"/>\n' + extra + "\n".join(cfg_parts) + "\n  </object>")
        cfg_inst.append(f'    <model_instance>\n      <metadata key="object_id" value="{oid}"/>\n'
                        f'      <metadata key="instance_id" value="0"/>\n'
                        f'      <metadata key="identify_id" value="{100 + oi}"/>\n    </model_instance>')
        asm.append(f'   <assemble_item object_id="{oid}" instance_id="0" transform="{t}" offset="0 0 0" />')
    files["3D/3dmodel.model"] = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {ns}>'
        '<metadata name="Application">BambuStudio-01.09.07.52</metadata>'
        '<metadata name="BambuStudio:3mfVersion">1</metadata>'
        f'<metadata name="Title">{title}</metadata>'
        f'<resources>{"".join(main_objs)}</resources><build p:UUID="{U()}">{"".join(build_items)}</build></model>')
    files["3D/_rels/3dmodel.model.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>")
    files["Metadata/model_settings.config"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "\n".join(cfg_objs) +
        '\n  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value="Capstan Arm mini"/>\n'
        '    <metadata key="locked" value="false"/>\n' + "\n".join(cfg_inst) + "\n  </plate>\n"
        "  <assemble>\n" + "\n".join(asm) + "\n  </assemble>\n</config>\n")
    files["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="png" ContentType="image/png"/></Types>')
    files["_rels/.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)


# --------------------------------------------------------------- main -------
def main():
    scale = float(eval(sys.argv[1])) if len(sys.argv) > 1 else SCALE
    os.makedirs(os.path.join(OUT, "stl"), exist_ok=True)
    for f in os.listdir(os.path.join(OUT, "stl")):
        os.remove(os.path.join(OUT, "stl", f))
    groups, cut_tool, Fs = build(scale)
    print("booleans ...")
    pieces, dowels = finish(groups, cut_tool, Fs, scale)

    # assembled view (as the arm stands)
    asm_meshes = []
    for name, *_ in PIECES:
        for s, reg in pieces[name]["regions"].items():
            if not reg.is_empty():
                tm = to_trimesh(reg)
                asm_meshes.append(((tm.vertices, tm.faces), SLOT_RGB[s]))
    render(None, os.path.join(OUT, "mini_assembled.png"), view=(1, -1.1, 0.6), meshes=asm_meshes,
           size=(1400, 1050))
    render(None, os.path.join(OUT, "mini_assembled_back.png"), view=(-1, 1.0, 0.5), meshes=asm_meshes,
           size=(1400, 1050))

    # lay out for printing
    placed, objects, stats = [], [], []
    x_cursor, y_cursor, row_h = 10.0, 10.0, 0.0
    for name, title, _ in PIECES:
        P = pieces[name]
        solid_tm = to_trimesh(P["solid"])
        Rm = print_orientation(name, P["solid"], cut_tool)
        Rm = best_spin(solid_tm, Rm)
        v = solid_tm.vertices @ Rm.T
        lo, hi = v.min(0), v.max(0)
        w, d = hi[0] - lo[0], hi[1] - lo[1]
        if x_cursor + w > BED - 10:
            x_cursor, y_cursor, row_h = 10.0, y_cursor + row_h + 10, 0.0
        Pm = np.eye(4)
        Pm[:3, :3] = Rm
        Pm[:3, 3] = [x_cursor - lo[0], y_cursor - lo[1], -lo[2]]
        x_cursor += w + 10
        row_h = max(row_h, d)
        parts = []
        for s in SLOTS:
            reg = P["regions"][s]
            if reg.is_empty() or reg.volume() < 1e-3:
                continue
            tm = to_trimesh(reg)
            tm.export(os.path.join(OUT, "stl", f"{name}_{s}.stl"))
            parts.append((f"{name}_{s}", s, tm))
        objects.append((name, parts, Pm))
        stats.append(dict(name=name, title=title, size=hi - lo, vol=P["solid"].volume() / 1000,
                          ovh=overhang_area(solid_tm, Rm), dropped=P["dropped"],
                          vols={s: P["regions"][s].volume() / 1000 for s in SLOTS}))
    # centre the whole layout
    allv = np.vstack([(np.c_[tm.vertices, np.ones(len(tm.vertices))] @ Pm.T)[:, :3]
                      for _, parts, Pm in objects for *_, tm in parts])
    shift = BED / 2 - (allv.min(0)[:2] + allv.max(0)[:2]) / 2
    for _, _, Pm in objects:
        Pm[:2, 3] += shift
    assert allv.max(0)[0] - allv.min(0)[0] < BED - 4 and allv.max(0)[1] - allv.min(0)[1] < BED - 4, "plate overflow"
    write_bambu_3mf(os.path.join(OUT, "capstan_arm_mini.3mf"), objects, "capstan_arm_mini")

    plate_meshes = []
    for _, parts, Pm in objects:
        for _, s, tm in parts:
            t = tm.copy()
            t.apply_transform(Pm)
            plate_meshes.append(((t.vertices, t.faces), SLOT_RGB[s]))
    bed = trimesh.creation.box((BED, BED, 0.6))
    bed.apply_translation((BED / 2, BED / 2, -0.3))
    plate_meshes.append(((bed.vertices, bed.faces), (0.80, 0.82, 0.85)))
    render(None, os.path.join(OUT, "mini_plate.png"), view=(0.25, -1, 0.75), meshes=plate_meshes, size=(1400, 1050),
           zoom=1.35)

    # report
    tot = sum(s["vol"] for s in stats)
    grams = {s: sum(st["vols"][s] for st in stats) * 1.27 * 0.55 for s in SLOTS}
    md = [f"# Capstan Arm 5, miniature ({FULL_HEIGHT * scale:.0f} mm / {FULL_HEIGHT * scale / 25.4:.1f} in tall, 1:{1 / scale:.1f})", "",
          "A display model of the whole arm in its display pose, in the real colours, sized for a desk. "
          "The outside matches the full-size arm; the inside is solid.", "",
          "![assembled](mini_assembled.png)", "",
          "## Print", "",
          "Open `capstan_arm_mini.3mf` in Bambu Studio. It's one plate with four multi-colour pieces, and each "
          "piece's colours are already assigned to filaments 1–4. Load the AMS to match:", "",
          "| AMS slot | Filament | ≈ g |", "|---:|---|---:|"]
    for i, s in enumerate(SLOTS, 1):
        md.append(f"| {i} | PETG or PLA, {s} ({SLOT_HEX[s]}) | {grams[s]:.0f} |")
    md += ["",
           "- **Settings:** 0.12 mm layers (0.08 mm if you want the screw heads crisp), 3 walls, 15 % infill."
           + (" At this size a 0.2 mm nozzle keeps much more detail (sector spokes, screw heads, gripper links); "
              "the standard 0.4 mm nozzle works but softens the small features." if scale < 0.35 else ""),
           "- **Supports and brims are already set** on each piece in the project: tree (auto) supports in graphite for "
           "B–D (the actuators stick out sideways), plus a brim on C and D because they stand on their small cut faces.",
           "- **Purge:** enable *flush into objects' infill* to cut purge waste; most colour changes happen inside solid infill.",
           "- PLA works as well as PETG here and is easier at this scale.",
           "- If the colours don't come through (an older Bambu Studio), run `python make_mini.py` to get per-colour STLs in `stl/` (left out of the download to keep it small): select "
           "all of one piece's files, import them together, answer **Yes** to \"load as a single object with "
           "multiple parts\", then set each part's filament.",
           "", "| Piece | What | Size on the bed (mm) | Prints on |", "|---|---|---|---|"]
    on = {"A_base": "its flat bottom", "B_shoulder": "the turret drum",
          "C_elbow": "the upper-arm cut face", "D_wrist": "the forearm cut face"}
    for st in stats:
        sz = st["size"]
        md.append(f"| {st['name'][0]} | {st['title']} | {sz[0]:.0f} × {sz[1]:.0f} × {sz[2]:.0f} | {on[st['name']]} |")
    md += ["", "## Assemble", "",
           "The pieces glue together; where there's room, a short length of 1.75 mm filament in a 2.0 mm hole "
           "works as a dowel to line them up.", ""]
    for joint, where, n, L in dowels:
        md.append(f"- **{joint}** ({'yaw axis' if where == 'J1 axis' else where.replace('upper', 'upper-arm cut').replace('fore', 'forearm cut')}): "
                  + (f"{n} × {L:.0f} mm dowel{'s' if n > 1 else ''}." if n else
                   "no dowel. The link is too thin for one at this size, but the two faces are the same shape, "
                   "so line up the edges and glue flat."))
    md += ["", "Glue with CA, or with a drop of 2-part epoxy if you want time to line things up. Dry-fit first. "
           "The cut faces only meet one way. The base–turret joint can also be left unglued so the "
           "turret swivels.", "",
           f"Regenerate with `python make_mini.py` (or e.g. `python make_mini.py 1/3` for a smaller one). "
           f"It rebuilds from the same parametric source, so any change to the arm shows up in the mini.", ""]
    open(os.path.join(OUT, "MINI.md"), "w").write("\n".join(md))
    for st in stats:
        print(st["name"], np.round(st["size"], 1), f"vol {st['vol']:.1f} cm3", f"overhang {st['ovh']:.0f} mm2",
              f"dropped {st['dropped']:.2f} mm3", {k: round(v, 2) for k, v in st["vols"].items()})
    print("dowels", dowels, "grams", {k: round(v) for k, v in grams.items()})


if __name__ == "__main__":
    main()
