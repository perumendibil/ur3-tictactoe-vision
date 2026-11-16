"""Tests for gripper.py with a fake Arduino. Run from robot/ur:  python -m unittest -v"""

import unittest

from gripper import Gripper, GripperError


class FakeArduino:
    """Answers like gripper_controller.ino (Hampus), without the delays."""

    def __init__(self, activated=True, dead=False):
        self.activated, self.dead, self.position, self.out, self.seen = activated, dead, 0, [], []

    def write(self, data):
        command = data.decode().strip().upper()
        self.seen.append(command)
        if self.dead:
            return
        if command == "PING":
            self.out += ["OK"]
        elif command == "STATUS":
            state = "INACTIVE NONE" if not self.activated else (
                "OPEN 0" if self.position == 0 else f"POSITIONED {self.position}")
            self.out += ["OK", "STATE " + state]
        elif command == "ACTIVATE":
            self.activated, self.position = True, 0
            self.out += ["OK", "DONE"]
        elif not self.activated:
            self.out += ["ERROR INACTIVE"]
        elif command.startswith("SET ") and command[4:].isdigit() and int(command[4:]) <= 100:
            self.position = int(command[4:])
            self.out += ["OK", "DONE"]
        else:
            self.out += ["ERROR INVALID_POSITION" if command.startswith("SET ") else "ERROR UNKNOWN_COMMAND"]

    def reset_input_buffer(self):
        self.out = []

    def readline(self):
        return (self.out.pop(0) + "\r\n").encode() if self.out else b""  # b"" = timeout


def gripper(**kw):
    arduino = FakeArduino(**kw)
    return Gripper(None, 50, 90, link=arduino), arduino


class GripperTest(unittest.TestCase):
    def test_open_and_close_send_the_calibrated_values(self):
        g, arduino = gripper()
        g.open()
        g.close()
        self.assertEqual(arduino.seen, ["SET 50", "SET 90"])
        self.assertEqual(arduino.position, 90)
        self.assertEqual(arduino.out, [])  # every reply line was read

    def test_ping_and_status(self):
        g, _ = gripper()
        self.assertEqual(g.command("PING"), "OK")
        self.assertTrue(g.is_active())

    def test_inactive_is_reported_and_moving_is_an_error(self):
        g, _ = gripper(activated=False)
        self.assertFalse(g.is_active())
        with self.assertRaises(GripperError):
            g.close()
        g.activate()
        self.assertTrue(g.is_active())
        g.close()

    def test_no_answer_is_an_error(self):
        g, _ = gripper(dead=True)
        with self.assertRaises(GripperError):
            g.open()

    def test_a_late_reply_does_not_answer_the_next_command(self):
        g, arduino = gripper()
        arduino.out = ["DONE"]  # left over from an older command
        self.assertEqual(g.command("STATUS"), "STATE OPEN 0")

    def test_bad_position_is_an_error(self):
        g, _ = gripper()
        g.close_set = 150
        with self.assertRaises(GripperError):
            g.close()


if __name__ == "__main__":
    unittest.main()
