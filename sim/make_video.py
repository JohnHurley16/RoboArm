"""Video: the arm picks the floppy beanbag off the table and throws it at the
board -- the same open-loop throw from three different pickups, to show how
much the grip matters (this is what the RL policy has to learn to correct).

    python sim/make_video.py          -> sim/out/pick_and_throw.mp4
"""
import json, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, mujoco, imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
from skills import Sim, pick, throw
from scene import Bag, Board
from pick_eval import random_pose
import throw_study as T

W, H, FPS = 1280, 720, 30
EVERY = int(round(1 / FPS / Sim.dt))          # record one physics state per video frame (real time)


def font(sz, bold=False):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def label(img, title, sub=""):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, H - 86, W, H], fill=(20, 22, 26))
    d.text((32, H - 78), title, fill=(245, 245, 245), font=font(32, True))
    d.text((32, H - 36), sub, fill=(245, 140, 60), font=font(21))
    return np.asarray(im)


def board_frame(board, p):
    """(along the slope from the front edge, across, height above the top) for a world point."""
    th = board.tilt
    dx, dz = p[0] - board.distance, p[2] - board.front_h
    return dx * math.cos(th) + dz * math.sin(th), p[1], -dx * math.sin(th) + dz * math.cos(th)


class Scorer:
    """Cornhole scoring like the training env: 3 if the bag passes down through
    the hole, else 1 if it ends resting on the board."""
    def __init__(self, board):
        self.b, self.through = board, False
        self.s_hole = board.length - 9 * 0.0254 * board.scale

    def __call__(self, sim):
        if sim.pinched >= 0 or self.through:
            return
        a, c, up = board_frame(self.b, sim.bag_pos())
        hs = self.b.hole_d * math.sqrt(math.pi) / 2 / 2           # the modelled hole: equal-area square
        if up < -0.005 and abs(a - self.s_hole) < hs and abs(c) < hs:
            self.through = True

    def points(self, sim):
        if self.through:
            return 3
        a, c, up = board_frame(self.b, sim.bag_pos())
        # resting on the board, including sagging over the hole's edge (real rules: 1 point)
        on = 0 < a < self.b.length and abs(c) < self.b.width / 2 and -2 * sim.bag.thick < up < 0.03
        return 1 if on else 0


def run(board, seed, params, record=True):
    rng = np.random.default_rng(seed)
    sim = Sim(Bag(), board=board, bag_pose=random_pose(rng), lean=not record, record_every=EVERY if record else 0)
    ok, _ = pick(sim, pose_noise=0.004, rng=rng)
    n_pick = len(sim.frames) if record else 0
    if not ok:
        return sim, n_pick, None
    sc = Scorer(board)
    throw(sim, params, on_step=sc)
    return sim, n_pick, sc.points(sim)


def main():
    study = json.load(open(os.path.join(T.OUT, "throw_study.json")))
    params = study["mini 2 in, 25 g"]["params"]
    # put the board where a typical pickup's throw lands in the hole
    land_x = study["grasp_scatter"]["mean_xy"][0]
    b0 = Board(scale=1 / 3, distance=2.0)
    s_hole = b0.length - 9 * 0.0254 * b0.scale
    # the bag crosses the board's height ~10 cm before it would reach the floor
    board = Board(scale=1 / 3, distance=land_x - 0.10 - s_hole * math.cos(b0.tilt))
    seeds = [s for s in range(40)][:3]
    frames = []
    r = m = d = None
    for n_run, seed in enumerate(seeds, 1):
        sim, n_pick, pts = run(board, seed, params)
        if r is None:                       # one renderer for all runs (the scenes are identical)
            m, d = sim.m, mujoco.MjData(sim.m)
            r = mujoco.Renderer(m, H, W)
        hole = np.array(board.hole_center())
        for i, q in enumerate(sim.frames):
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()
            if i < n_pick:                       # close on the pickup
                cam.lookat[:] = [0.18, 0.0, 0.08]; cam.distance, cam.azimuth, cam.elevation = 0.75, 150, -22
                title, sub = f"Throw {n_run} of 3: pick the bag off the table", "press the pads onto the bag and pinch: the beans squeeze aside, the pads clamp the cloth"
            elif i < len(sim.frames) - int(1.2 * FPS):  # steady side view of the throw and flight
                cam.lookat[:] = [hole[0] * 0.5, 0.0, 0.32]; cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -10
                title = f"Throw {n_run} of 3: same open-loop throw"
                sub = "the floppy bag sits a little differently in the pinch every time"
            else:                                        # close on the board for the result
                cam.lookat[:] = hole + [0, 0, 0.0]; cam.distance, cam.azimuth, cam.elevation = 0.95, 60, -28
                title = f"Throw {n_run} of 3: same open-loop throw"
                sub = {3: "IN THE HOLE: 3 points", 1: "on the board: 1 point", 0: "missed", None: "dropped"}[pts]
            r.update_scene(d, cam)
            frames.append(label(r.render(), title, sub))
        frames += [frames[-1]] * FPS
        print(f"throw {n_run}: points {pts}, bag ended at {np.round(sim.bag_pos(), 2)}, hole {np.round(board.hole_center(), 2)}", flush=True)
    out = os.path.join(HERE, "out", "pick_and_throw.mp4")
    iio.mimwrite(out, frames, fps=FPS, quality=8, macro_block_size=8)
    print(out, f"{len(frames) / FPS:.1f} s")


if __name__ == "__main__":
    main()
