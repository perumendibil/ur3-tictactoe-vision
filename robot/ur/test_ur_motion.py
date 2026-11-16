"""Tests for ur_motion.py, without a robot. Run from robot/ur:  python -m unittest -v"""

import unittest

import ur_motion as m

POS = m.load_positions()


class FakeUR(m.UR):
    """A UR that never touches the network."""

    def __init__(self, ready=True, arrives=True):
        super().__init__("0.0.0.0")
        self.sent, self._ready, self.arrives = [], ready, arrives

    def _send(self, text):
        self.sent.append(text)

    def dashboard(self, command):
        return "fake " + command

    def ready(self):
        return self._ready

    def state(self):
        if self.arrives and self.sent:
            return list(POS["photo_joints_deg"]), [300.95, -100.04, 32.1]
        return [0] * 6, [262.0, -114.3, 298.2]


def moves(program):
    return [line.strip() for line in program.splitlines() if "movel" in line or "movej" in line]


class Programs(unittest.TestCase):
    def test_positions_file(self):
        self.assertEqual(sorted(POS["cells_mm"]), [str(c) for c in range(1, 10)])
        self.assertEqual(len(POS["store_o_mm"]), 5)
        self.assertEqual(len(POS["photo_joints_deg"]), 6)

    def test_pick_from_photo_pose_goes_straight_over_the_hole(self):
        lines = moves(m.pick_program(POS, 0, [262.0, -114.3, 298.2]))
        self.assertEqual(lines, [
            "movel(p[0.36101, -0.23537, 0.08500, t[3], t[4], t[5]], a=0.2, v=0.08)",
            "movel(p[0.36101, -0.23537, 0.03991, t[3], t[4], t[5]], a=0.2, v=0.08, r=0.004)",
            "movel(p[0.36101, -0.23537, 0.02991, t[3], t[4], t[5]], a=0.2, v=0.01)",
        ])

    def test_pick_from_low_rises_first(self):
        lines = moves(m.pick_program(POS, 0, [300.0, -100.0, 40.0]))
        self.assertEqual(lines[0], "movel(p[0.30000, -0.10000, 0.08500, t[3], t[4], t[5]], a=0.2, v=0.08)")
        self.assertEqual(len(lines), 4)

    def test_carry(self):
        lines = moves(m.carry_program(POS, 1, 5))
        self.assertEqual(lines, [
            "movel(p[0.32410, -0.27983, 0.05000, t[3], t[4], t[5]], a=0.2, v=0.01, r=0.004)",
            "movel(p[0.32410, -0.27983, 0.08500, t[3], t[4], t[5]], a=0.2, v=0.08)",
            "movel(p[0.30095, -0.10004, 0.08500, t[3], t[4], t[5]], a=0.2, v=0.08)",
            "movel(p[0.30095, -0.10004, 0.04210, t[3], t[4], t[5]], a=0.2, v=0.08, r=0.004)",
            "movel(p[0.30095, -0.10004, 0.03210, t[3], t[4], t[5]], a=0.2, v=0.01)",
        ])

    def test_photo_with_lift(self):
        lines = moves(m.photo_program(POS))
        self.assertEqual(lines[0], "movel(p[t[0], t[1], 0.08500, t[3], t[4], t[5]], a=0.2, v=0.08, r=0.02)")
        self.assertTrue(lines[1].startswith("movej([0.00541, -1.48039,"))

    def test_photo_without_lift_is_one_line(self):
        self.assertEqual(m.photo_program(POS, lift=False).count("\n"), 1)

    def test_last_move_of_each_program_has_no_blend(self):
        # The wait loop trusts that the robot stops at the last point.
        for program in (m.pick_program(POS, 2, [0, 0, 298]), m.carry_program(POS, 2, 9)):
            self.assertNotIn("r=", moves(program)[-1])

    def test_pose_variable_is_not_r(self):
        for program in (m.pick_program(POS, 0, [0, 0, 298]), m.carry_program(POS, 0, 1),
                        m.photo_program(POS)):
            self.assertNotIn("r[", program)


class Robot(unittest.TestCase):
    def test_not_ready_sends_nothing(self):
        ur = FakeUR(ready=False)
        with self.assertRaises(m.RobotError):
            ur.carry(POS, 0, 5)
        self.assertEqual(ur.sent, [])

    def test_arrives(self):
        ur = FakeUR()
        ur.carry(POS, 0, 5)
        self.assertEqual(len(ur.sent), 1)

    def test_timeout_stops_the_robot(self):
        ur = FakeUR(arrives=False)
        with self.assertRaises(m.RobotError):
            ur.go_to(m.carry_program(POS, 0, 5), 300.95, -100.04, 32.1, timeout_s=0.3)
        self.assertEqual(ur.sent[-1], "stopl(0.5)\n")

    def test_lost_pose_stops_the_robot(self):
        ur = FakeUR()
        ur.state = lambda: (_ for _ in ()).throw(OSError("port 30003 closed"))
        with self.assertRaises(m.RobotError):
            ur.go_to(m.carry_program(POS, 0, 5), 300.95, -100.04, 32.1)
        self.assertEqual(ur.sent[-1], "stopl(0.5)\n")


if __name__ == "__main__":
    unittest.main()
