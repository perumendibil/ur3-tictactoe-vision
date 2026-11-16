"""The gripper: a servo on an Arduino UNO R4 WiFi running Hampus's firmware.

Firmware: https://github.com/Hampusman/ArduinoGripper (gripper_controller.ino).
USB serial at 115200 baud, one command per line. Replies:
    PING            -> OK
    STATUS          -> OK, then STATE INACTIVE NONE | STATE OPEN 0 | STATE POSITIONED 50 ...
    ACTIVATE        -> OK, then DONE   (sweeps the servo 0-180 deg for ~5 s: gripper EMPTY)
    SET <0-100>     -> OK, then DONE   (0 = fully open, 100 = fully closed)
    anything wrong  -> ERROR INACTIVE | ERROR INVALID_POSITION | ERROR UNKNOWN_COMMAND

DONE only means "the time to move has passed": the firmware cannot feel a block.
The director checks with the camera that the piece really arrived.
"""

import time

BAUD = 115200
LAST_LINES = ("DONE", "ERROR", "STATE")  # a reply ends with one of these (PING: with OK)


class GripperError(RuntimeError):
    pass


class Gripper:
    def __init__(self, port, open_set, close_set, link=None, timeout_s=10):
        """port: "COM3" on Peru's laptop. link: an already open serial port (for the tests)."""
        self.open_set, self.close_set = int(open_set), int(close_set)
        if link is None:
            import serial  # pyserial; here so the tests run without it
            # pyserial raises DTR when it opens the port: without DTR the R4 does not answer.
            link = serial.Serial(port, BAUD, timeout=timeout_s)
            time.sleep(0.5)
            link.reset_input_buffer()
        self.link = link

    def command(self, text):
        """Send one command and return its last reply line. Raises GripperError."""
        self.link.reset_input_buffer()  # a late reply to an older command must not answer this one
        self.link.write((text + "\n").encode())
        while True:
            line = self.link.readline().decode(errors="replace").strip()
            if not line:
                raise GripperError(f"no answer to {text!r} (cable? Serial Monitor open?)")
            if line.startswith("ERROR"):
                raise GripperError(f"{text!r} -> {line}")
            if line.startswith(LAST_LINES) or (line == "OK" and text == "PING"):
                return line

    def is_active(self):
        return self.command("STATUS") != "STATE INACTIVE NONE"

    def activate(self):
        """Only with the gripper empty and hands away: the servo sweeps its whole range."""
        self.command("ACTIVATE")

    def open(self):
        self.command(f"SET {self.open_set}")

    def close(self):
        self.command(f"SET {self.close_set}")

    def release_port(self):
        self.link.close()
