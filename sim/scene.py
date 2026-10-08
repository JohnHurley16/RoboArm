"""Scene XML: the arm plus floor, lights, beanbag and (optionally) a cornhole board.

Everything is parametric so tasks and the throw study can build the scene they need:

    xml = scene_xml(bag=Bag(size=0.05, mass=0.025), board=Board(scale=0.25, distance=1.5))
    model = mujoco.MjModel.from_xml_string(xml)

Units are metres and kg.  The floor (z = 0) is the surface the arm stands on.
"""
import json
import math
import os
import re
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
META = json.load(open(os.path.join(ASSETS, "arm_meta.json")))
BITS = META["bits"]
IN = 0.0254
PINCH_BODIES = ("bag", "bag_px", "bag_nx", "bag_py", "bag_ny", "bag_pxpy", "bag_pxny", "bag_nxpy", "bag_nxny")


@dataclass
class Bag:
    """Beanbag.  Regulation: 6 in square, about 1 in thick, 15-16 oz.
    Tabletop sets: about 2 in square and 20-30 g.

    model="floppy" (default): a 3 x 3 grid of small weighted spheres joined by
    hinges.  The hinges have a little spring, damping and dry friction: dry
    friction is what makes beans in fabric stay where they're pushed instead of
    springing back, so the bag drapes over the gripper, flops in flight and
    slumps when it lands.  Spheres keep the contacts cheap on the GPU (MJX has
    no soft bodies).  Tune bend_* with sim/bag_calibrate.py against a real bag.
    model="rigid": a single soft box (the old model)."""
    size: float = 0.050        # side, m
    thick: float = 0.016       # m
    mass: float = 0.025        # kg
    friction: float = 0.8
    model: str = "floppy"
    bend_stiffness: float = 0.0006    # N m / rad per hinge (weak: a bag only slowly flattens itself)
    bend_damping: float = 0.00012     # N m s / rad
    bend_friction: float = 0.0012     # N m (dry friction: holds folds)
    twist_scale: float = 2.0          # twist joints are this much stiffer than bend joints
    hinge_armature: float = 2e-6      # kg m^2: keeps the tiny beads numerically stable under hard impacts
    # beanbags barely bounce: soft, over-damped contact
    solref: str = "0.008 1.6"
    solimp: str = "0.9 0.95 0.003"

    @property
    def n_tiles(self):
        return 9 if self.model == "floppy" else 1


def _bag_excludes(bag: Bag):
    """Beads that touch the same fold line but aren't directly linked: no contact
    between them, or they prop the bag up instead of letting it fold."""
    if bag.model != "floppy":
        return ""
    pairs = [("bag_pxpy", "bag_py"), ("bag_pxny", "bag_ny"), ("bag_nxpy", "bag_py"), ("bag_nxny", "bag_ny"),
             ("bag", "bag_pxpy"), ("bag", "bag_pxny"), ("bag", "bag_nxpy"), ("bag", "bag_nxny"),
             ("bag_px", "bag_py"), ("bag_px", "bag_ny"), ("bag_nx", "bag_py"), ("bag_nx", "bag_ny"),
             ("bag_px", "bag_nx"), ("bag_py", "bag_ny")]
    return "".join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in pairs)


def _bag_xml(bag: Bag, pos):
    col = (f'contype="{BITS["BAG"]}" conaffinity="{BITS["WORLD"] | BITS["BAG"] | BITS["BASE"] | BITS["HAND"] | BITS["PAD"]}" '
           f'friction="{bag.friction} 0.02 0.005" condim="6" solref="{bag.solref}" solimp="{bag.solimp}"')
    if bag.model == "rigid":
        h = (bag.size / 2, bag.size / 2, bag.thick / 2)
        I = [bag.mass / 12 * (a * a + b * b) for a, b in ((2 * h[1], 2 * h[2]), (2 * h[0], 2 * h[2]), (2 * h[0], 2 * h[1]))]
        return (f'<body name="bag" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="bag"/>'
                f'<inertial pos="0 0 0" mass="{bag.mass}" diaginertia="{I[0]:.3e} {I[1]:.3e} {I[2]:.3e}"/>'
                f'<geom name="bag" type="box" size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}" material="bag" {col}/></body>')
    p = bag.size / 3                          # tile pitch
    r = min(bag.thick / 2, p / 2)             # bead radius
    m = bag.mass / 9
    I = 0.4 * m * r * r
    vis = f'type="box" size="{p / 2 * 0.98:.4f} {p / 2 * 0.98:.4f} {bag.thick / 2 * 0.9:.4f}" material="bag" contype="0" conaffinity="0" group="1" mass="0"'
    j = lambda name, axis, at, k: (f'<joint name="{name}" type="hinge" axis="{axis}" pos="{at}" range="-1.75 1.75" armature="{bag.hinge_armature}" '
                                   f'stiffness="{bag.bend_stiffness * k:.6f}" damping="{bag.bend_damping * k:.6f}" '
                                   f'frictionloss="{bag.bend_friction * k:.6f}"/>')

    fab = (f'type="box" size="{p / 2:.4f} {p / 2:.4f} 0.0010" rgba="0.75 0.12 0.10 0" mass="0" '
           f'contype="{BITS["FABRIC"]}" conaffinity="{BITS["PAD"]}" friction="{bag.friction} 0.02 0.005" condim="4"')

    def tile(name, extra=""):
        # bead (beans: collides with the world, the bag and the hand) + fabric patch (what the pads pinch)
        return (f'<inertial pos="0 0 0" mass="{m:.6f}" diaginertia="{I:.3e} {I:.3e} {I:.3e}"/>'
                f'<geom name="{name}" type="sphere" size="{r:.4f}" rgba="0.75 0.12 0.10 0" {col}/>'
                f'<geom name="{name}_vis" {vis}/>{extra}')

    kids = []
    for sx, sy, tag in ((1, 0, "px"), (-1, 0, "nx"), (0, 1, "py"), (0, -1, "ny")):
        # edge tile: bends about the in-plane axis across the link, twists about the link
        bend = f"{-sy} {sx} 0" if sx else f"{sy} 0 0"
        link = f"{sx} {sy} 0"
        at = f"{-sx * p / 2:.4f} {-sy * p / 2:.4f} 0"
        corners = ""
        if sx:   # x-edge tiles carry the corner tiles (tree: centre -> x-edges -> corners)
            for cy in (1, -1):
                ctag = f"{tag}{'py' if cy > 0 else 'ny'}"
                corners += (f'<body name="bag_{ctag}" pos="0 {cy * p:.4f} 0">'
                            + j(f"bag_{ctag}_b", f"{cy} 0 0", f"0 {-cy * p / 2:.4f} 0", 1.0)
                            + j(f"bag_{ctag}_t", f"0 1 0", f"0 {-cy * p / 2:.4f} 0", bag.twist_scale)
                            + tile(f"bag_{ctag}") + "</body>")
        kids.append(f'<body name="bag_{tag}" pos="{sx * p:.4f} {sy * p:.4f} 0">'
                    + j(f"bag_{tag}_b", bend, at, 1.0) + j(f"bag_{tag}_t", link, at, bag.twist_scale)
                    + tile(f"bag_{tag}", corners) + "</body>")
    return (f'<body name="bag" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="bag"/>'
            + tile("bag") + "".join(kids) + "</body>")


@dataclass
class Board:
    """Cornhole board.  scale=1 is regulation: 24 x 48 in top, 6 in hole whose
    centre is 9 in from the back edge, front edge 3 in high and back edge 12 in.
    distance is from the J1 axis to the board's front edge (regulation
    pitch is 27 ft = 8.23 m front edge to front edge)."""
    scale: float = 0.25
    distance: float = 1.5      # m
    yaw: float = 0.0           # deg, direction of the board from the arm (0 = +x)
    friction: float = 0.45     # bag on painted wood slides a bit

    @property
    def width(self):
        return 24 * IN * self.scale

    @property
    def length(self):
        return 48 * IN * self.scale

    @property
    def hole_d(self):
        return 6 * IN * self.scale

    @property
    def front_h(self):
        return 3 * IN * self.scale

    @property
    def back_h(self):
        return 12 * IN * self.scale

    @property
    def tilt(self):
        return math.atan2(self.back_h - self.front_h, self.length)

    def hole_center(self):
        """World position of the hole centre (on the top surface)."""
        s = self.length - 9 * IN * self.scale             # along the slope from the front edge
        x = self.distance + s * math.cos(self.tilt)
        z = self.front_h + s * math.sin(self.tilt)
        c, sn = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        return (x * c, x * sn, z)


def _board_xml(b: Board):
    """Board as a static body tilted about its front edge.  The top is four
    boxes around a square hole of the same area as the round one."""
    # top surface stays where it is; the collision slab extends 25 mm down so a fast bag
    # (about 1 cm per 2 ms physics step) can't tunnel through a thin scaled-down top
    t = max(0.012 * max(b.scale, 0.25), 0.025)
    W, L = b.width, b.length
    hs = b.hole_d * math.sqrt(math.pi) / 2         # equal-area square side
    hc = L - 9 * IN * b.scale                      # hole centre along the top from the front
    y0, y1 = -hs / 2, hs / 2
    x0, x1 = hc - hs / 2, hc + hs / 2
    boxes = [  # (cx, cy, hx, hy) in the board's top frame (x along the slope, from the front edge)
        (x0 / 2, 0, x0 / 2, W / 2),                                   # front of the hole
        ((x1 + L) / 2, 0, (L - x1) / 2, W / 2),                        # behind the hole
        (hc, (y0 - W / 2) / 2, hs / 2, (W / 2 + y0) / 2),             # left of the hole
        (hc, (y1 + W / 2) / 2, hs / 2, (W / 2 - y1) / 2),             # right of the hole
    ]
    # the board only needs to stop the bag (the arm never reaches it): keeps contact checks cheap
    cls = f'contype="{BITS["WORLD"]}" conaffinity="{BITS["BAG"]}" friction="{b.friction} 0.01 0.001" condim="4"'
    g = []
    for i, (cx, cy, hx, hy) in enumerate(boxes):
        g.append(f'<geom name="board_top{i}" type="box" pos="{cx:.4f} {cy:.4f} {-t / 2:.4f}" '
                 f'size="{hx:.4f} {hy:.4f} {t / 2:.4f}" material="board" {cls}/>')
    # side rails and front skirt so a bag can't slide under
    rail = 0.02 * max(b.scale, 0.25)
    for side in (-1, 1):
        g.append(f'<geom name="board_rail{side + 1}" type="box" pos="{L / 2:.4f} {side * (W / 2 - rail / 2):.4f} {-t - rail:.4f}" '
                 f'size="{L / 2:.4f} {rail / 2:.4f} {rail:.4f}" material="board_frame" {cls}/>')
    c, s = math.cos(math.radians(b.yaw)), math.sin(math.radians(b.yaw))
    px, py = b.distance * c, b.distance * s
    # front skirt (vertical, from the floor to the front edge)
    skirt = (f'<geom name="board_skirt" type="box" pos="{px:.4f} {py:.4f} {b.front_h / 2:.4f}" '
             f'euler="0 0 {math.radians(b.yaw):.5f}" size="{0.005:.4f} {W / 2:.4f} {b.front_h / 2:.4f}" material="board_frame" {cls}/>')
    # back panel (real boards are closed boxes: a bag through the hole stays under the board)
    g.append(f'<geom name="board_back" type="box" pos="{L:.4f} 0 {-b.back_h / 2 / math.cos(b.tilt):.4f}" '
             f'euler="0 {b.tilt:.5f} 0" size="0.005 {W / 2:.4f} {b.back_h / 2:.4f}" material="board_frame" {cls}/>')
    hx, hy, hz = b.hole_center()
    return f"""
    {skirt}
    <body name="board" pos="{px:.4f} {py:.4f} {b.front_h:.4f}" euler="0 {-b.tilt:.5f} {math.radians(b.yaw):.5f}">
      {chr(10).join('      ' + x for x in g)}
    </body>
    <site name="hole" pos="{hx:.4f} {hy:.4f} {hz:.4f}" size="{b.hole_d / 2:.4f}" rgba="0 0 0 0"/>"""


def scene_xml(bag: Bag | None = Bag(), board: Board | None = None, target_site=False, lean=True,
              bag_pos=(0.30, 0.25, 0.05)):
    """Full scene.  lean=True uses the collision-only arm (training);
    lean=False includes the visual meshes (rendering)."""
    arm = open(os.path.join(ASSETS, "arm_col.xml" if lean else "arm.xml")).read()
    arm = arm.replace('meshdir="."', f'meshdir="{ASSETS}"')
    floor = (f'<geom name="floor" type="plane" size="0 0 0.05" material="floor" '
             f'contype="{BITS["WORLD"]}" conaffinity="{BITS["BAG"] | BITS["HAND"] | BITS["PAD"]}" friction="0.7 0.01 0.001" condim="4"/>')
    # (arm links vs floor is checked geometrically in the envs, which is far cheaper than box contacts)
    extra = [
        '<light name="key" pos="1.2 -2.0 4" dir="-0.2 0.4 -1" directional="true" diffuse="0.6 0.6 0.6" castshadow="true"/>',
        '<light name="fill" pos="-1.5 1 2.5" dir="0.5 -0.3 -1" directional="true" diffuse="0.3 0.3 0.33" castshadow="false"/>',
        floor,
    ]
    if bag is not None:
        extra.append(_bag_xml(bag, bag_pos))
    if board is not None:
        extra.append(_board_xml(board))
    if target_site:
        extra.append('<body name="target" mocap="true" pos="0.3 0 0.2"><site name="target" size="0.012" rgba="0.1 0.9 0.3 0.6"/></body>')
    assets = """
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.70 0.71 0.73" rgb2="0.62 0.63 0.66"/>
    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.0"/>
    <material name="bag" rgba="0.75 0.12 0.10 1"/>
    <material name="board" rgba="0.92 0.86 0.72 1"/>
    <material name="board_frame" rgba="0.55 0.42 0.28 1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.75 0.78 0.84" width="256" height="256"/>"""
    arm = arm.replace("<asset>", "<asset>" + assets, 1)
    arm = arm.replace('<quality shadowsize="4096"/>', '<quality shadowsize="4096"/><headlight ambient="0.22 0.22 0.22" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>', 1)
    # attach the extras to the world body (before the arm's closing tag)
    i = arm.rfind("</worldbody>")
    arm = arm[:i] + "\n    " + "\n    ".join(extra) + "\n  " + arm[i:]
    if bag is not None and bag.model == "floppy":
        arm = arm.replace("</mujoco>", f"  <contact>{_bag_excludes(bag)}</contact>\n</mujoco>")
    if bag is not None:
        # pinch latches (sim/pinch.py): one inactive weld per bead, switched on when the pads squeeze the bag
        welds = "".join(f'<weld name="pinch_{b}" body1="grip" body2="{b}" active="false" solref="0.01 1" torquescale="0.3"/>'
                        for b in PINCH_BODIES[:bag.n_tiles])
        if "<equality>" in arm:
            arm = arm.replace("<equality>", "<equality>" + welds, 1)
        else:
            arm = arm.replace("</mujoco>", f"  <equality>{welds}</equality>\n</mujoco>")
    arm = arm.replace('<compiler angle="radian"', '<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>\n  <compiler angle="radian"', 1)
    return arm
