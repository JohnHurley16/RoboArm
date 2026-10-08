"""Calibrate and sanity-check the floppy beanbag model.

1. Drape: hold the bag by one edge (and by its middle) and measure how far the
   rest sags.  Do the same with a real bag (photo from the side, measure the
   angle) and adjust Bag.bend_* in sim/scene.py until they match.
2. Landing: toss the bag onto a board at a typical cornhole impact
   (4 m/s, 35 deg down) and compare bounce and slide with MuJoCo's full
   soft-body (flex) model on CPU.  The GPU can't run flex, so this checks that
   the bead-grid bag lands like a soft body.

    python sim/bag_calibrate.py            -> numbers + sim/out/bag_drape.png
"""
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                       # noqa: E402
import numpy as np                                  # noqa: E402

from scene import Bag, _bag_xml, _bag_excludes, BITS   # noqa: E402

FLOOR = (f'<geom name="floor" type="plane" size="2 2 .1" contype="{BITS["WORLD"]}" conaffinity="{BITS["BAG"]}" '
         f'friction="0.45 0.01 0.001" condim="4" rgba=".7 .7 .72 1"/>')
LIGHT = '<light pos="0 -1 2" dir="0 .5 -1" directional="true"/>'
ASSET = '<asset><material name="bag" rgba="0.75 0.12 0.10 1"/></asset>'


def drape(bag: Bag, hold="edge", seconds=2.0):
    """Weld one tile to the world and let gravity drape the rest.
    Returns the droop angle (deg below horizontal) of the far side."""
    held = "bag_nx" if hold == "edge" else "bag"
    far = "bag_px"
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="0.001" integrator="implicitfast"/>{ASSET}<worldbody>{LIGHT}
      {_bag_xml(bag, (0, 0, 0.3))}</worldbody><contact>{_bag_excludes(bag)}</contact>
      <equality><weld body1="{held}" solref="0.002 1"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    a, b = d.body(held).xpos, d.body(far).xpos
    v = b - a
    return math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1]))), m, d


def toss(xml_body, names, v=4.0, angle=35.0, seconds=1.5, dt=0.001):
    """Throw a bag onto a flat surface; returns bounce height and slide distance."""
    integ = "discrete" if "flexcomp" in xml_body else "implicitfast"
    exc = "" if "flexcomp" in xml_body else f"<contact>{_bag_excludes(Bag())}</contact>"
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="{dt}" integrator="{integ}" cone="elliptic" impratio="10"/>{ASSET}
      <worldbody>{LIGHT}{FLOOR}{xml_body}</worldbody>{exc}</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    vx, vz = v * math.cos(math.radians(angle)), -v * math.sin(math.radians(angle))
    d.qvel[:] = 0
    if m.nflex:
        d.qvel[0::3], d.qvel[2::3] = vx, vz            # flex vertices: 3 dofs each
    else:
        d.qvel[0], d.qvel[2] = vx, vz
    mujoco.mj_forward(m, d)
    com = lambda: (np.mean(d.flexvert_xpos, axis=0) if m.nflex else d.subtree_com[m.body("bag").id]).copy()
    p0 = None
    zmin_after, hmax = None, 0.0
    for _ in range(int(seconds / dt)):
        mujoco.mj_step(m, d)
        c = com()
        if p0 is None and d.ncon > 0:
            p0 = c.copy()                               # first touch
        if p0 is not None:
            hmax = max(hmax, c[2] - p0[2])
    return dict(bounce_mm=hmax * 1000, slide_mm=(com()[0] - p0[0]) * 1000 if p0 is not None else float("nan"))


def flex_bag(bag: Bag, young=3e4):
    n = 4
    sp = bag.size / (n - 1)
    return (f'<flexcomp name="flexbag" type="grid" count="{n} {n} 2" spacing="{sp:.4f} {sp:.4f} {bag.thick:.4f}" '
            f'pos="0 0 {bag.thick / 2 + 0.06:.3f}" radius="0.004" dim="3" mass="{bag.mass}" rgba=".75 .12 .1 1">'
            f'<contact condim="4" friction="{bag.friction} 0.02 0.005" solref="{bag.solref}" solimp="{bag.solimp}" '
            f'contype="{BITS["BAG"]}" conaffinity="{BITS["WORLD"]}" selfcollide="none"/>'
            f'<edge damping="0.02"/><elasticity young="{young}" poisson="0.2" damping="0.002"/></flexcomp>')


def main():
    bag = Bag()
    for hold in ("edge", "middle"):
        ang, m, d = drape(bag, hold)
        print(f"held by its {hold:6s}: far side droops {ang:5.1f} deg below horizontal")
        if hold == "edge":
            try:
                r = mujoco.Renderer(m, 360, 480)
                cam = mujoco.MjvCamera()
                cam.lookat[:] = d.body("bag").xpos
                cam.distance, cam.azimuth, cam.elevation = 0.18, 90, 0
                r.update_scene(d, cam)
                from PIL import Image
                Image.fromarray(r.render()).save(os.path.join(HERE, "out", "bag_drape.png"))
            except Exception as e:                       # noqa: BLE001
                print("  (no render:", e, ")")
    beads = toss(_bag_xml(bag, (0, 0, bag.thick / 2 + 0.06)), None)
    flex = toss(flex_bag(bag), None, dt=0.0002)
    print(f"toss at 4 m/s, 35 deg down:  bead-grid bag bounce {beads['bounce_mm']:.1f} mm, slide {beads['slide_mm']:.0f} mm"
          f"   |   flex soft body bounce {flex['bounce_mm']:.1f} mm, slide {flex['slide_mm']:.0f} mm")


if __name__ == "__main__":
    main()
