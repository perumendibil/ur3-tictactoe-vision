"""Move the UR3 (CB3, PolyScope 3.15) from the PC.

URScript programs go to port 30002; the PC knows the robot has arrived by reading
its pose on port 30003, and checks robot mode and safety on port 29999 (Dashboard).
Decision 2026-11-10 in DECISIONES.md: port 30002 now, ur_rtde later.

Units: positions.json and every function here use mm and degrees; the URScript
text gets metres and radians. The tool keeps the orientation it has when each
program starts (as in the lab helpers of 2026-11-10).

One turn = three programs, because the gripper is driven by the PC (COM3) and
the robot must be still at each grip:
    pick_program  : above the store hole -> down to grip height
    carry_program : up with the piece -> over the square -> down to release height
    photo_program : straight up to travel height -> photo pose, without stopping
"""

import json
import math
import os
import socket
import struct
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS = os.path.join(HERE, "posiciones.json")

FAST, SLOW = 0.08, 0.01      # m/s: travel, and the last/first mm next to a piece
ACC = 0.2                    # m/s^2
SLOW_ZONE_MM = 10            # moving down: last 10 mm at SLOW
LIFT_SLOW_TO_MM = 50         # moving up with a piece: SLOW until this height
VERTICAL_BLEND_M = 0.004     # blend on the same vertical line: only removes the stop
PHOTO_BLEND_M = 0.02         # rise -> photo pose: the turn starts ~20 mm below travel height
REACHED_MM, REACHED_DEG = 1.0, 0.05


def load_positions(path=POSITIONS):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# URScript text (no robot needed; tested in test_ur_motion.py)
# --------------------------------------------------------------------------

def _program(name, waypoints):
    """waypoints: (x_mm, y_mm, z_mm, speed_m_s, blend_m). Orientation from the start pose.

    The local is called t, not r: r= is already the blend parameter of movel.
    """
    lines = [f"def {name}():", "  t = get_actual_tcp_pose()"]
    for x, y, z, v, blend in waypoints:
        extra = f", r={blend}" if blend else ""
        lines.append(f"  movel(p[{x / 1000:.5f}, {y / 1000:.5f}, {z / 1000:.5f}, t[3], t[4], t[5]],"
                     f" a={ACC}, v={v}{extra})")
    lines.append("end")
    return "\n".join(lines) + "\n"


def pick_program(pos, hole, now_mm):
    """From anywhere (usually the photo pose) down to grip height over store hole 0-4."""
    sx, sy, sz = pos["store_o_mm"][hole]
    travel = pos["travel_z_mm"]
    points = []
    if now_mm[2] < travel - REACHED_MM:      # below travel height: first straight up
        points.append((now_mm[0], now_mm[1], travel, FAST, 0))
    points += [(sx, sy, travel, FAST, 0),
               (sx, sy, sz + SLOW_ZONE_MM, FAST, VERTICAL_BLEND_M),
               (sx, sy, sz, SLOW, 0)]
    return _program("director_pick", points)


def carry_program(pos, hole, cell):
    """From grip height in the store (piece held) down to release height over square 1-9.

    It stops above the square before going down (no blend at that corner yet:
    the block height is not measured; DECISIONES.md 2026-11-10).
    """
    sx, sy, _ = pos["store_o_mm"][hole]
    cx, cy = pos["cells_mm"][str(cell)]
    travel = pos["travel_z_mm"]
    release = pos["board_grip_z_mm"] + pos["release_above_grip_mm"]
    return _program("director_carry", [
        (sx, sy, LIFT_SLOW_TO_MM, SLOW, VERTICAL_BLEND_M),
        (sx, sy, travel, FAST, 0),
        (cx, cy, travel, FAST, 0),
        (cx, cy, release + SLOW_ZONE_MM, FAST, VERTICAL_BLEND_M),
        (cx, cy, release, SLOW, 0),
    ])


def photo_program(pos, lift=True):
    """Photo pose. lift=True: first straight up to travel height, blended into the movej."""
    q = ", ".join(f"{math.radians(d):.5f}" for d in pos["photo_joints_deg"])
    move = f"movej([{q}], a=0.5, v=0.4)"
    if not lift:
        return move + "\n"
    return ("def director_photo():\n  t = get_actual_tcp_pose()\n"
            f"  movel(p[t[0], t[1], {pos['travel_z_mm'] / 1000:.5f}, t[3], t[4], t[5]],"
            f" a={ACC}, v={FAST}, r={PHOTO_BLEND_M})\n"  # gripper open: no piece to drop
            f"  {move}\nend\n")


# --------------------------------------------------------------------------
# Talking to the robot
# --------------------------------------------------------------------------

class RobotError(RuntimeError):
    pass


class UR:
    def __init__(self, ip, timeout_s=3):
        self.ip = ip
        self.timeout_s = timeout_s

    def _send(self, text):
        with socket.create_connection((self.ip, 30002), timeout=self.timeout_s) as s:
            s.sendall(text.encode())

    def dashboard(self, command):
        with socket.create_connection((self.ip, 29999), timeout=self.timeout_s) as s:
            s.recv(200)  # welcome line
            s.sendall((command + "\n").encode())
            return s.recv(200).decode().strip()

    def ready(self):
        return ("RUNNING" in self.dashboard("robotmode")
                and "NORMAL" in self.dashboard("safetymode"))

    def state(self):
        """(joints in degrees, TCP x, y, z in mm) from the real-time port 30003."""
        with socket.create_connection((self.ip, 30003), timeout=self.timeout_s) as s:
            data = b""
            while len(data) < 1220:  # offsets checked on this robot on 2026-11-10
                chunk = s.recv(4096)
                if not chunk:
                    raise RobotError("port 30003 closed the connection")
                data += chunk
        if struct.unpack(">i", data[:4])[0] < 492:  # first field = packet length
            raise RobotError("port 30003 sent an unexpected packet")
        joints = struct.unpack(">6d", data[252:300])
        tcp = struct.unpack(">6d", data[444:492])
        return [math.degrees(a) for a in joints], [v * 1000 for v in tcp[:3]]

    def run(self, program, reached, timeout_s):
        """Send a program and wait until reached(joints_deg, tcp_mm) is True.

        If it does not arrive in time, or the pose cannot be read, it stops the
        robot (stopl) and raises.
        """
        if not self.ready():
            raise RobotError("robot not ready: " + self.dashboard("robotmode")
                             + " | " + self.dashboard("safetymode"))
        self._send(program)
        start = time.time()
        try:
            while time.time() - start < timeout_s:
                if reached(*self.state()):
                    return
                time.sleep(0.2)
            problem = f"target not reached in {timeout_s} s"
        except (OSError, RobotError) as err:
            problem = f"lost the robot pose ({err})"
        self.stop()
        raise RobotError(problem + ": stop sent to the robot")

    def stop(self):
        """Stop the arm now: a new script on port 30002 replaces the running program."""
        self._send("stopl(0.5)\n")

    def go_to(self, program, x_mm, y_mm, z_mm, timeout_s=60):
        self.run(program, lambda _j, p: all(abs(a - b) < REACHED_MM
                                            for a, b in zip(p, (x_mm, y_mm, z_mm))), timeout_s)

    def go_photo(self, pos, lift=True, timeout_s=60):
        target = pos["photo_joints_deg"]
        self.run(photo_program(pos, lift),
                 lambda j, _p: max(abs(a - b) for a, b in zip(j, target)) < REACHED_DEG, timeout_s)
        time.sleep(1)  # let the arm settle so the photo is not blurred

    # The two piece moves, with the target each program must reach.
    def pick(self, pos, hole):
        sx, sy, sz = pos["store_o_mm"][hole]
        self.go_to(pick_program(pos, hole, self.state()[1]), sx, sy, sz)

    def carry(self, pos, hole, cell):
        cx, cy = pos["cells_mm"][str(cell)]
        self.go_to(carry_program(pos, hole, cell), cx, cy,
                   pos["board_grip_z_mm"] + pos["release_above_grip_mm"])
