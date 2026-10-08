"""Small assembly framework: parts with a design key, a colour and a placement."""
import math
import numpy as np
import cadquery as cq
from OCP.gp import gp_Trsf

COLORS = {
    "printed": (0.92, 0.92, 0.90),     # white / light grey PETG (links, shells)
    "printed2": (0.19, 0.20, 0.23),    # graphite PETG (joints, actuators)
    "accent": (0.96, 0.42, 0.10),      # orange PETG (capstan drums, pulleys)
    "bearing": (0.72, 0.74, 0.78),
    "steel": (0.60, 0.62, 0.66),
    "screw": (0.15, 0.15, 0.17),
    "brass": (0.80, 0.62, 0.25),
    "motor": (0.12, 0.12, 0.14),
    "motor_bell": (0.55, 0.10, 0.12),
    "copper": (0.78, 0.45, 0.20),
    "pcb": (0.10, 0.45, 0.25),
    "magnet": (0.55, 0.55, 0.60),
    "cable": (0.98, 0.85, 0.15),
}


def mat_loc(M):
    """4x4 numpy matrix -> cq.Location."""
    t = gp_Trsf()
    t.SetValues(*[float(M[i, j]) for i in range(3) for j in range(4)])
    return cq.Location(t)


def T(x=0.0, y=0.0, z=0.0):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def R(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Rm = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    M = np.eye(4)
    M[:3, :3] = Rm
    return M


def frame(xdir, ydir, zdir, origin=(0, 0, 0)):
    """Matrix whose local X,Y,Z axes map to the given world directions."""
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = xdir, ydir, zdir
    M[:3, 3] = origin
    assert abs(np.linalg.det(M[:3, :3]) - 1) < 1e-6, "not a proper rotation"
    return M


class Part:
    def __init__(self, key, shape, kind="printed", color=None, M=None,
                 bom=None, qty_note=None, print_down=None):
        """
        key        unique design id (one STL per printed key)
        kind       printed | hardware | motor | cable | pcb
        bom        purchased-part description (hardware) used for counting
        print_down for printed parts: unit vector (part frame) that should face
                   the print bed
        """
        self.key = key
        self.shape = shape
        self.kind = kind
        self.color = color or COLORS["printed" if kind == "printed" else "steel"]
        self.M = np.eye(4) if M is None else M
        self.bom = bom
        self.print_down = print_down

    def placed(self, M):
        p = Part(self.key, self.shape, self.kind, self.color, M @ self.M,
                 self.bom, print_down=self.print_down)
        return p

    def world_shape(self):
        s = self.shape.val() if isinstance(self.shape, cq.Workplane) else self.shape
        if isinstance(self.shape, cq.Workplane) and len(self.shape.vals()) > 1:
            s = cq.Compound.makeCompound(self.shape.vals())
        return s.moved(mat_loc(self.M))

    def local_shape(self):
        if isinstance(self.shape, cq.Workplane):
            vals = self.shape.vals()
            return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
        return self.shape


def place_all(parts, M):
    return [p.placed(M) for p in parts]


# ----------------------------------------------------------- 2.5D helpers ---
def to_link_plane(wp):
    """Shapes built in local XY (X=link x, Y=link z) and extruded along local
    Z are mapped so that local Z -> world -y.  So build with Z in [-y1, -y0]
    to occupy world y in [y0, y1]."""
    return wp.rotate((0, 0, 0), (1, 0, 0), 90)


def slab(sketch_fn, y0, y1):
    """Extrude a 2D outline (function returning a Workplane sketch on a given
    workplane) to occupy world y in [y0, y1] in the link plane."""
    wp = cq.Workplane("XY").workplane(offset=-y1)
    solid = sketch_fn(wp).extrude(y1 - y0)
    return to_link_plane(solid)


def cyl_y(x, z, r, y0, y1):
    """Cylinder along y at link-plane point (x, z)."""
    c = (cq.Workplane("XY").workplane(offset=-y1).center(x, z)
         .circle(r).extrude(y1 - y0))
    return to_link_plane(c)


def hull_pts(circles, rects=(), n=48):
    """Convex hull outline of circles (cx, cy, r) and rects (x0, y0, x1, y1)."""
    pts = []
    for cx, cy, r in circles:
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for x0, y0, x1, y1 in rects:
        pts += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    from scipy.spatial import ConvexHull
    P = np.array(pts)
    h = ConvexHull(P)
    return [tuple(P[i]) for i in h.vertices]


# ------------------------------------------------- smooth convex outlines ---
def circle_hull(circles):
    """Exact convex hull of circles [(cx, cy, r)] as an ordered list of
    segments: ('arc', c, a_in, a_out) and ('line', p0, p1), CCW."""
    from scipy.spatial import ConvexHull
    circles = [(float(a), float(b), float(c)) for a, b, c in circles]
    if len(circles) == 1:
        return [("circle", circles[0])]
    pts, tag = [], []
    for i, (cx, cy, r) in enumerate(circles):
        for k in range(360):
            a = 2 * math.pi * k / 360
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            tag.append(i)
    P = np.array(pts)
    h = ConvexHull(P)
    seq = []
    for v in h.vertices:
        if not seq or seq[-1] != tag[v]:
            seq.append(tag[v])
    while len(seq) > 1 and seq[0] == seq[-1]:
        seq.pop()
    if len(seq) == 1:
        return [("circle", circles[seq[0]])]
    tangents = []
    n = len(seq)
    for k in range(n):
        c1, c2 = circles[seq[k]], circles[seq[(k + 1) % n]]
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        L = math.hypot(dx, dy)
        th = math.atan2(dy, dx)
        al = math.acos(max(-1.0, min(1.0, (c1[2] - c2[2]) / L)))
        nx, ny = math.cos(th - al), math.sin(th - al)
        tangents.append(((c1[0] + c1[2] * nx, c1[1] + c1[2] * ny),
                         (c2[0] + c2[2] * nx, c2[1] + c2[2] * ny), math.atan2(ny, nx)))
    segs = []
    for k in range(n):
        c = circles[seq[(k + 1) % n]]
        a_in = tangents[k][2]
        a_out = tangents[(k + 1) % n][2]
        segs.append(("line", tangents[k][0], tangents[k][1]))
        if c[2] > 1e-6:
            segs.append(("arc", c, a_in, a_out))
    return segs


def hull_sketch(circles):
    """Returns fn(wp) -> wp with a closed smooth hull wire (for slab())."""
    segs = circle_hull(circles)

    def fn(wp):
        if segs[0][0] == "circle":
            cx, cy, r = segs[0][1]
            return wp.center(cx, cy).circle(r)
        start = segs[0][1]
        w = wp.moveTo(*start)
        for s in segs:
            if s[0] == "line":
                w = w.lineTo(*s[2])
            else:
                (cx, cy, r), a0, a1 = s[1], s[2], s[3]
                if a1 < a0:
                    a1 += 2 * math.pi
                if a1 - a0 < 1e-6:
                    continue
                am = (a0 + a1) / 2
                w = w.threePointArc((cx + r * math.cos(am), cy + r * math.sin(am)),
                                    (cx + r * math.cos(a1), cy + r * math.sin(a1)))
        return w.close()
    return fn


def rrect(x0, y0, x1, y1, r):
    """Rounded rect as 4 circles (for circle_hull)."""
    assert x1 - x0 >= 2 * r - 1e-9 and y1 - y0 >= 2 * r - 1e-9, "rrect smaller than its corner radius"
    return [(x0 + r, y0 + r, r), (x1 - r, y0 + r, r), (x1 - r, y1 - r, r), (x0 + r, y1 - r, r)]


def smooth_slab(circles, y0, y1, edge=1.5, kind="fillet"):
    """Hull-of-circles plate in the link plane, y in [y0, y1], soft edges."""
    lo, hi = min(y0, y1), max(y0, y1)
    wp = cq.Workplane("XY").workplane(offset=-hi)
    solid = hull_sketch(circles)(wp).extrude(hi - lo)
    if edge:
        try:
            s2 = solid.faces(">Z or <Z").edges()
            s2 = s2.fillet(edge) if kind == "fillet" else s2.chamfer(edge)
            if s2.val().isValid():
                solid = s2
        except Exception:
            try:
                solid = solid.faces(">Z or <Z").edges().chamfer(edge * 0.7)
            except Exception:
                pass
    return to_link_plane(solid)


def floor_slab(circles, z0, z1, edge=1.5, top_only=False):
    wp = cq.Workplane("XY").workplane(offset=z0)
    solid = hull_sketch(circles)(wp).extrude(z1 - z0)
    if edge:
        try:
            sel = ">Z" if top_only else ">Z or <Z"
            s2 = solid.faces(sel).edges().fillet(edge)
            if s2.val().isValid():
                solid = s2
        except Exception:
            pass
    return solid
