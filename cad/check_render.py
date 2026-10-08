import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # paths below are relative to cad/
_os.makedirs("out", exist_ok=True)
import sys, time
from arm.arm import *
from arm.render import render, world_meshes
B = build_static()
for name, q in (("pose", POSE), ("home", HOME)):
    parts = assemble(B, q)
    W = world_meshes(parts)
    render(None, f"check/arm_{name}.png", view=(1.0, -1.6, 0.7), meshes=W, size=(1400, 1100))
    if name == "home":
        render(None, "check/arm_home_side.png", view=(0, -1, 0), meshes=W, size=(1600, 900), ortho=True)
        render(None, "check/arm_home_top.png", view=(0, 0, 1), up=(1, 0, 0), meshes=W, size=(1600, 900), ortho=True)
        render(None, "check/arm_home_back.png", view=(0.9, 1.2, 0.5), meshes=W, size=(1400, 1100))
