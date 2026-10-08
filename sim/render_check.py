import math, os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco, numpy as np
from PIL import Image
m = mujoco.MjModel.from_xml_path("sim/assets/arm.xml")
d = mujoco.MjData(m)
POSE = dict(J1=25.0, J2=55.0, J3=-85.0, J4=-30.0, J5=30.0, G=25.0)
def setq(q):
    for k in ("J1","J2","J3","J4","J5","G"):
        d.joint(k).qpos = math.radians(q[k])
    g = math.radians(q["G"])
    for k, s in (("Gl",-1),("Gku",-1),("Gkl",1),("Gfu",1),("Gfl",-1)):
        d.joint(k).qpos = s*g
    mujoco.mj_forward(m, d)
setq(POSE)
r = mujoco.Renderer(m, 600, 800)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.15, 0.05, 0.15]; cam.distance = 0.95; cam.azimuth = -125; cam.elevation = -18
opt = mujoco.MjvOption()
imgs = []
for show_col in (False, True):
    opt.geomgroup[3] = show_col; opt.geomgroup[1] = True
    r.update_scene(d, cam, opt); imgs.append(r.render())
Image.fromarray(np.hstack(imgs)).save("sim/out/model_check.png")
print("tcp", d.site("tcp").xpos)
sys.path.insert(0, "ik"); import capstan_ik as K
print("ik fk", K.Arm().fk(POSE))
