"""Move the real arm to a tool pose: IK -> moteus position commands.

Untested on hardware (written before the arm exists).  Run it first with the
cables slack or the arm supported, and low limits, until the motor signs and
offsets in capstan_ik.py are confirmed.

    pip install moteus numpy
    python drive_example.py 300 0 80 --pitch -60 --roll 0 --grip 25
"""
import argparse
import asyncio
import math

import moteus

import capstan_ik as K

VEL_LIMIT = 2.0      # rotor rev/s  (J2 at 10.4:1 -> about 70 deg/s at the joint)
ACC_LIMIT = 4.0      # rotor rev/s^2
MAX_TORQUE = 0.6     # Nm at the motor; keep low while testing


def send_servo(deg):
    """Gripper servo.  Replace with whatever your Pico firmware accepts, e.g.
    serial.Serial('/dev/ttyACM0').write(f'S {deg:.1f}\\n'.encode())."""
    print(f"servo -> {deg:.1f} deg")


async def move_to(q_motor, transport, ctrls, timeout=10.0):
    """Command every joint at once, then poll until all have arrived."""
    t0 = asyncio.get_event_loop().time()
    while True:
        cmds = [ctrls[k].make_position(position=q_motor[k], velocity=0.0, velocity_limit=VEL_LIMIT,
                                       accel_limit=ACC_LIMIT, maximum_torque=MAX_TORQUE,
                                       watchdog_timeout=math.nan, query=True)
                for k in K.JOINTS]
        res = await transport.cycle(cmds)
        pos = {K.JOINTS[i]: r.values[moteus.Register.POSITION] for i, r in enumerate(res)}
        err = max(abs(pos[k] - q_motor[k]) for k in K.JOINTS)
        if err < 0.005:                       # 0.005 rotor rev ~ 0.2-0.3 deg at the joint
            return pos
        if asyncio.get_event_loop().time() - t0 > timeout:
            raise TimeoutError(f"not there after {timeout} s, worst error {err:.3f} rev")
        await asyncio.sleep(0.02)


async def main():
    ap = argparse.ArgumentParser()
    for a in ("x", "y", "z"):
        ap.add_argument(a, type=float)
    ap.add_argument("--pitch", type=float, default=-60.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--grip", type=float, default=20.0)
    a = ap.parse_args()

    arm = K.Arm()
    transport = moteus.get_singleton_transport()
    ctrls = {k: moteus.Controller(id=K.CAN_ID[k]) for k in K.JOINTS}

    # where are we now?  (seed keeps the solver on the current branch)
    res = await transport.cycle([ctrls[k].make_query() for k in K.JOINTS])
    now = K.Arm.from_motors({K.JOINTS[i]: r.values[moteus.Register.POSITION] for i, r in enumerate(res)})

    sol = arm.ik(a.x, a.y, a.z, a.pitch, a.roll, a.grip, seed=now, strict=True)
    print(sol)
    await move_to(sol.motors, transport, ctrls)
    send_servo(sol.motors["servo"])


if __name__ == "__main__":
    asyncio.run(main())
